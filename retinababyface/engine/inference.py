import os
import math
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union, Optional, Callable

import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
from matplotlib.patches import Polygon as MplPolygon
from torch.utils.data import DataLoader
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
)

from engine.train import (
    infer_with_rotated_nms,
    get_resize_size,
    generate_anchors_for_training,
    xyxyxyxy2xywhr,
    batch_probiou,
    denormalize_image,
)
from data_setup.augmentations import wrap_to_pi

# -----------------------------------------------------------------------------
# I. Model Checkpoint and Anchor Preparation
# -----------------------------------------------------------------------------


def load_model_checkpoint(
    model: torch.nn.Module, path: str, device: torch.device
) -> None:
    """
    Loads the model weights from a checkpoint file and prepares it for inference.

    Args:
        model (torch.nn.Module): The model to load the weights into.
        path (str): Path to the checkpoint file (.pth or .pt).
        device (torch.device): The device to load the model onto.
    """
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get(
        "model_state_dict", checkpoint
    )  # support plain or wrapped checkpoints
    model.load_state_dict(state_dict)
    model.to(device).eval()  # set model to evaluation mode
    logging.info(f"Model checkpoint loaded from {path}")


def prepare_anchors(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    scale_factors: List[float],
    ratio_factors: List[float],
    anchors_cache_path: str,
) -> Tuple[Tuple[int, int], torch.Tensor, torch.Tensor]:
    """
    Prepares anchors for inference using the base OBB statistics and given scale/ratio factors.

    Args:
        model (torch.nn.Module): The model used for feature extraction to generate anchors.
        loader (DataLoader): DataLoader to estimate image resize size.
        device (torch.device): Device for tensor creation (usually 'cuda' or 'cpu').
        scale_factors (List[float]): List of scaling factors for anchor generation.
        ratio_factors (List[float]): List of aspect ratio factors for anchors.
        anchors_cache_path (str): Path to cache the generated anchors for reuse.
    Returns:
        Tuple containing:
            - resize_size (Tuple[int, int]): Target size used to resize images.
            - anchors_xy (torch.Tensor): Anchors in (x, y) format for initial location.
            - anchors_xywhr (torch.Tensor): Anchors in (x, y, w, h, θ) format.
    """
    # Get target resize size from the dataset
    resize_size = get_resize_size(loader)

    # Generate anchors in both formats
    anchors_xy, anchors_xywhr = generate_anchors_for_training(
        model=model,
        resize_size=resize_size,
        device=device,
        scale_factors=scale_factors,
        ratio_factors=ratio_factors,
        anchors_cache_path=anchors_cache_path,
    )

    print(
        f"[INFO] Generated {anchors_xy.shape[0]} anchors for image size {resize_size}"
    )
    logging.info(f"✅ Anchors prepared successfully")
    return resize_size, anchors_xy, anchors_xywhr


# -----------------------------------------------------------------------------
# II. Inference Loop & Data Accumulation
# -----------------------------------------------------------------------------


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    anchors_xy: torch.Tensor,
    resize_size: Tuple[int, int],
    face_thres: float,
    iou_thres: float,
    class_thres: float,
    baby_thres: float,
    device: torch.device,
    labels_map: Dict[int, str],
    render_original: bool = False,
) -> Dict[str, Any]:
    """
    Runs inference on a dataset and collects predictions, ground truths, and metrics.

    Args:
        - model (torch.nn.Module): The trained model to use for inference.
        - loader (DataLoader): DataLoader providing the test dataset.
        - anchors_xy (torch.Tensor): Precomputed anchors in (x, y) format.
        - resize_size (Tuple[int, int]): Target size used to resize images.
        - face_thres (float): Confidence threshold for face detection.
        - iou_thres (float): IoU threshold for matching predictions to ground truths.
        - class_thres (float): Class score threshold for filtering predictions.
        - baby_thres (float): Baby face confidence threshold for filtering predictions.
        - device (torch.device): Device to run inference on (e.g., 'cuda' or 'cpu').
        - labels_map (Dict[int, str]): Mapping from class indices to human-readable labels.
        - render_original (bool): Whether to render original images instead of normalized ones.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - per_true (Dict[int, List[int]]): Binary ground truth (1 for TP, 0 for FP) per class.
            - per_score (Dict[int, List[float]]): Confidence scores of predictions per class.
            - iou_errs (Dict[int, List[float]]): IoU errors per class.
            - angle_errs (Dict[int, List[float]]): Angle errors per class.
            - stats (Dict[int, Dict[str, int]]): TP, FP, FN counts per class.
            - y_true (List[int]): Ground truth labels for confusion matrix.
            - y_pred (List[int]): Predicted labels for confusion matrix.
            - all_gts (List[int]): Ground truth labels for F1 vs. threshold computation.
            - all_preds (List[int]): Predicted labels for F1 vs. threshold computation.
            - all_scores (List[float]): Prediction scores for F1 vs. threshold computation.
            - child_stats (Dict[str, int]): TP, FP, FN counts for child face detection.
            - child_gt (List[int]): Ground truth labels for child face detection (0 = adult, 1 = child).
            - child_pred (List[int]): Predicted labels for child face detection (0 = adult, 1 = child).
            - samples (List[Tuple[Any, Dict[str, torch.Tensor], str, torch.Tensor, torch.Tensor, torch.Tensor, int, int]]):
              List of qualitative samples for visualization, including images, predictions, ground truths, and error counts.
    """
    # Initialize data structures for metrics and qualitative results
    per_true = {c: [] for c in labels_map}
    per_score = {c: [] for c in labels_map}
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in labels_map}
    y_true, y_pred = [], []
    iou_errs = {c: [] for c in labels_map}
    angle_errs = {c: [] for c in labels_map}
    child_stats = {"tp": 0, "fp": 0, "fn": 0}
    samples = []

    # Additional lists for F1 vs. threshold computation
    all_gts = []  # Ground truth labels
    all_preds = []  # Predicted labels
    all_scores = []  # Prediction scores
    child_gt, child_pred = [], []

    def log_child(gt_is_baby: bool, pred_is_baby: bool):
        child_gt.append(1 if gt_is_baby else 0)
        child_pred.append(1 if pred_is_baby else 0)

    dataset = loader.dataset
    global_idx = 0

    model.eval()  # Set model to evaluation mode
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference"):
            imgs = batch["image"].to(device)
            targets = batch["target"]

            # Perform inference with rotated NMS
            outputs = infer_with_rotated_nms(
                model,
                imgs,
                anchors_xy,
                resize_size,
                face_thres,
                baby_thres,
                iou_thres,
                class_thres,
            )

            batch_size = imgs.size(0)
            for b in range(batch_size):
                fname = dataset.file_list[global_idx]
                global_idx += 1
                fp_img, fn_img = 0, 0

                viz_payload = None
                if render_original:
                    orig_img_np, (sx, sy) = _load_original_and_scale(dataset, fname, resize_size)
                    if orig_img_np is not None:
                        viz_payload = {"orig_img": orig_img_np, "scale": (sx, sy)}
                    else:
                        viz_payload = None

                # --------------------- Ground Truth Processing -------------------
                valid_mask = targets["valid_mask"][b]
                gt_boxes = targets["boxes"][b][valid_mask]  # GT boxes coordinates
                gt_angles = targets["angles"][b][valid_mask].view(
                    -1
                )  # GT rotation angles
                gt_labels = targets["class_idx"][b][valid_mask]  # GT class labels
                gt_child = targets["child_prob"][b][valid_mask] > 0.5  # Child face mask
                num_gt = gt_boxes.size(0)

                # Convert GT boxes to xywhr format for rotated IoU computation
                gt_xywhr = xyxyxyxy2xywhr(
                    gt_boxes, gt_angles.unsqueeze(-1), resize_size
                ).to(device)
                gt_matched = torch.zeros(num_gt, dtype=torch.bool, device=device)

                # --------------------- Model Predictions ------------------------
                pred_boxes = outputs[b]["boxes"].to(
                    device
                )  # Predicted boxes (N_pred, 5)
                pred_scores = outputs[b]["scores"].to(device)  # Confidence scores
                pred_labels = outputs[b]["labels"].to(device)  # Predicted class labels
                pred_child_s = outputs[b]["child_score"].to(
                    device
                )  # Adult/child scores
                pred_is_child = outputs[b]["is_child"].to(device)
                num_pred = pred_boxes.size(0)

                # ----------------- Handle Images without GT (num_gt==0) ----------
                if num_gt == 0:
                    for det_idx in range(num_pred):
                        cls_det = int(pred_labels[det_idx])
                        score_det = float(pred_scores[det_idx])

                        # Update PR/F1 metrics
                        per_true[cls_det].append(0)  # All predictions are FP
                        per_score[cls_det].append(score_det)
                        all_gts.append(-1)  # Background class
                        all_preds.append(cls_det)
                        all_scores.append(score_det)

                        # Update stats and confusion matrix
                        stats[cls_det]["fp"] += 1  # Count as False Positive
                        fp_img += 1
                        y_true.append(-1)  # Background row
                        y_pred.append(cls_det)  # Predicted class column

                        pred_baby = bool(
                            pred_is_child[det_idx].item()
                        )  # Is it a child face?
                        log_child(False, pred_baby)
                        # Update child stats
                        if bool(pred_is_child[det_idx].item()):
                            # Child face detected
                            child_stats["fp"] += 1

                    # Store qualitative sample
                    samples.append(
                        (
                            imgs[b].cpu(),
                            {k: v.cpu().detach() for k, v in outputs[b].items()},
                            fname,
                            gt_boxes.cpu(),
                            gt_angles.cpu(),
                            gt_labels.cpu(),
                            fp_img,
                            fn_img,
                        )
                    )
                    continue  # Next image
                # ----------------------------------------------------------------

                # Compute complete IoU matrix (num_gt × num_pred)
                iou_matrix = batch_probiou(gt_xywhr, pred_boxes)

                # Sort detections by confidence score (descending)
                _, idxs_det = torch.sort(pred_scores, descending=True)

                # Process each detection in order of confidence
                for det_idx in idxs_det.tolist():
                    score_det = float(pred_scores[det_idx])
                    cls_det = int(pred_labels[det_idx])

                    # Find best matching GT for this detection
                    unmatched_mask = ~gt_matched  # Unmatched GT mask
                    ious_all = iou_matrix[
                        :, det_idx
                    ].clone()  # IoUs with current detection
                    ious_all[~unmatched_mask] = -1  # Exclude already matched GTs

                    if unmatched_mask.any():
                        best_iou_val, best_gt_idx = ious_all.max(0)
                        best_gt_idx = int(best_gt_idx.item())
                        best_iou_val = float(best_iou_val.item())
                    else:  # All GTs matched → can only be FP
                        best_iou_val, best_gt_idx = -1.0, -1

                    if best_iou_val >= iou_thres:
                        true_cls = int(gt_labels[best_gt_idx])
                        gt_baby = bool(gt_child[best_gt_idx].item())
                        pred_baby = bool(pred_is_child[det_idx].item())
                        log_child(gt_baby, pred_baby)

                        if pred_baby and gt_baby:
                            child_stats["tp"] += 1
                        elif pred_baby and not gt_baby:
                            child_stats["fp"] += 1
                        elif (not pred_baby) and gt_baby:
                            child_stats["fn"] += 1

                        if cls_det == true_cls and true_cls in stats:
                            # -------------------- TRUE POSITIVE --------------------
                            stats[true_cls]["tp"] += 1
                            gt_matched[best_gt_idx] = True

                            # Update metrics
                            per_true[true_cls].append(1)
                            per_score[true_cls].append(score_det)
                            y_true.append(true_cls)
                            y_pred.append(true_cls)

                            # Compute geometric errors
                            iou_errs[true_cls].append(best_iou_val)
                            angle_diff = pred_boxes[det_idx, 4] - gt_angles[best_gt_idx]
                            angle_errs[true_cls].append(
                                float(wrap_to_pi(angle_diff).abs() * 180.0 / math.pi)
                            )

                            all_gts.append(true_cls)
                            all_preds.append(true_cls)
                            all_scores.append(score_det)

                        else:
                            # -------------- CLASS CONFUSION ERROR -----------------
                            # Wrong class but good localization
                            if cls_det in stats:
                                stats[cls_det]["fp"] += 1
                                fp_img += 1
                                per_true[cls_det].append(0)
                                per_score[cls_det].append(score_det)

                            if true_cls in stats:
                                stats[true_cls]["fn"] += 1
                                fn_img += 1
                                gt_matched[best_gt_idx] = True

                            # CM and global curves can include classes not in labels_map
                            y_true.append(true_cls)  # GT class row
                            y_pred.append(cls_det)  # Predicted class column
                            all_gts.append(true_cls)
                            all_preds.append(cls_det)
                            all_scores.append(score_det)

                    else:
                        # ---------------- BACKGROUND FALSE POSITIVE --------------
                        # No matching GT with sufficient IoU
                        if cls_det in stats:
                            stats[cls_det]["fp"] += 1
                            fp_img += 1
                            per_true[cls_det].append(0)
                            per_score[cls_det].append(score_det)
                            pred_baby = bool(pred_is_child[det_idx].item())
                            log_child(False, pred_baby)

                        if bool(pred_is_child[det_idx]):
                            child_stats["fp"] += 1

                        y_true.append(-1)  # Background row
                        y_pred.append(cls_det)  # Predicted class column

                        all_gts.append(-1)
                        all_preds.append(cls_det)
                        all_scores.append(score_det)

                # ---- Process unmatched GT boxes as False Negatives -------------
                for i in range(num_gt):
                    if not gt_matched[i]:
                        # GT box not matched to any detection
                        cls_gt = int(gt_labels[i])

                        # Update PR/F1 metrics
                        if cls_gt in labels_map:
                            per_true[cls_gt].append(1)  # True for GT
                            per_score[cls_gt].append(0.0)  # Score is 0 for unmatched GT
                            if cls_gt in stats:
                                stats[cls_gt]["fn"] += 1  # Count as False Negative
                                fn_img += 1
                            y_true.append(cls_gt)  # GT class row
                            y_pred.append(-1)  # Background column
                        else:
                            # GT with no original class (should not happen)
                            y_true.append(-1)
                            y_pred.append(-1)

                        # This three can include classes not in labels_map
                        all_gts.append(cls_gt)  # GT class
                        all_preds.append(-1)  # Background class
                        all_scores.append(0.0)  # Score is 0 for unmatched GT

                        # Update child/adult if needed
                        gt_baby = bool(gt_child[i].item())
                        log_child(gt_baby, False)
                        if gt_baby:
                            child_stats["fn"] += 1

                # ---- Store qualitative sample ----------------------------------
                samples.append(
                    (
                        imgs[b].cpu(),
                        {k: v.cpu().detach() for k, v in outputs[b].items()},
                        fname,
                        gt_boxes.cpu(),
                        gt_angles.cpu(),
                        gt_labels.cpu(),
                        fp_img,
                        fn_img,
                        viz_payload,
                    )
                )

            # Clean up to free memory
            del imgs, outputs, targets
            torch.cuda.empty_cache()

    # Finalize metrics for each class
    for cls in labels_map:
        # Ensure every class has at least one entry in per_true/per_score
        if not per_true[cls]:
            # If no predictions for this class, set to empty lists
            per_true[cls].append(0)
            # If no predictions for this class, set score to 0.0
            per_score[cls].append(0.0)

    print(f"[INFO] Inference completed on {global_idx} samples.")
    return {
        "per_true": per_true,
        "per_score": per_score,
        "iou_errs": iou_errs,
        "angle_errs": angle_errs,
        "stats": stats,
        "y_true": y_true,
        "y_pred": y_pred,
        "all_gts": all_gts,
        "all_preds": all_preds,
        "all_scores": all_scores,
        "child_stats": child_stats,
        "child_gt": child_gt,
        "child_pred": child_pred,
        "samples": samples,
    }


# -----------------------------------------------------------------------------
# III. Metric Computation & Plotting
# -----------------------------------------------------------------------------


def compute_map_and_pr(
    per_true: Dict[int, List[int]], per_score: Dict[int, List[float]]
) -> Tuple[float, Dict[int, float]]:
    """
    Computes the mean Average Precision (mAP) and per-class AP using precision-recall curves.

    Args:
        per_true (Dict[int, List[int]]): Binary ground truth (1 for TP, 0 for FN/FP) per class.
        per_score (Dict[int, List[float]]): Confidence scores of predictions per class.

    Returns:
        Tuple:
            - float: mean Average Precision across all classes.
            - Dict[int, float]: Average Precision per class.
    """
    APs = {
        cls: average_precision_score(per_true[cls], per_score[cls])
        if sum(per_true[cls]) > 0
        else 0.0
        for cls in per_true
    }
    mAP = float(np.mean(list(APs.values())))
    print(f"[INFO] Computed mAP: {mAP:.4f}")
    return mAP, APs


def smooth_curve(x: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Applies a Gaussian smoothing filter to a 1D array.

    Args:
        x (np.ndarray): Input array to smooth.
        sigma (float): Standard deviation of the Gaussian filter.

    Returns:
        np.ndarray: Smoothed array.
    """
    return gaussian_filter1d(x, sigma=sigma)


def plot_precision_recall(
    per_true: Dict[int, List[int]],
    per_score: Dict[int, List[float]],
    labels_map: Dict[int, str],
    mAP: float,
    sigma: float = 2.0,
) -> plt.Figure:
    """
    Plots a smoothed Precision-Recall (PR) curve per class and a global average.

    Args:
        per_true (Dict[int, List[int]]): Binary true labels per class.
        per_score (Dict[int, List[float]]): Prediction scores per class.
        labels_map (Dict[int, str]): Mapping from class index to label string.
        mAP (float): Mean Average Precision across all classes.
        sigma (float): Smoothing factor for curves.

    Returns:
        matplotlib.figure.Figure: PR curve figure.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Precision-Recall Curve", fontsize=13)

    classes = list(labels_map.keys())

    # Plot per-class PR curves
    for cls in classes:
        y_t = np.array(per_true[cls], dtype=int)
        y_s = np.array(per_score[cls], dtype=float)

        if y_t.sum() == 0:
            # Avoid division by zero: constant precision
            prec, rec = np.ones(10), np.linspace(0, 1, 10)
        else:
            prec, rec, _ = precision_recall_curve(y_t, y_s)

        # prec_s = smooth_curve(prec, sigma)
        # rec_s = smooth_curve(rec, sigma)
        ap = average_precision_score(y_t, y_s) if y_t.sum() > 0 else 0.0

        ax.plot(rec, prec, lw=2, label=f"{labels_map[cls]} {ap:.3f}")

    # Plot global PR curve
    all_true = np.concatenate([per_true[c] for c in classes])
    all_scores = np.concatenate([per_score[c] for c in classes])

    prec_all, rec_all, _ = precision_recall_curve(all_true, all_scores)
    prec_all_s = smooth_curve(prec_all, sigma)
    rec_all_s = smooth_curve(rec_all, sigma)
    ax.plot(
        rec_all_s,
        prec_all_s,
        lw=3,
        color="blue",
        label=f"all classes {mAP:.3f} mAP@0.5",
    )

    # Axes and styling
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1.0), fontsize=12, frameon=False)
    plt.tight_layout()
    print("[INFO] PR curve plotted.")
    return fig


def plot_confusion_matrix(
    y_true: List[int], y_pred: List[int], labels_map: Dict[int, str]
) -> Dict[str, plt.Figure]:
    """
    Plots both the raw and normalized confusion matrices.

    Args:
        y_true (List[int]): Ground truth class indices.
        y_pred (List[int]): Predicted class indices (may include -1 for background).
        labels_map (Dict[int, str]): Mapping from class index to class name.

    Returns:
        Dict[str, plt.Figure]: Dictionary with 'raw' and 'normalized' confusion matrix plots.
    """
    labels = list(labels_map.keys()) + [-1]
    names = [labels_map.get(l, "BG") for l in labels]

    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaNs from division by zero

    # Raw matrix plot
    fig_raw, ax_raw = plt.subplots(figsize=(6, 6))
    im_raw = ax_raw.imshow(cm_raw, cmap="Blues")
    for i in range(len(names)):
        for j in range(len(names)):
            val = cm_raw[i, j]
            if val == 0:
                continue
            text = (
                f"{np.diag(cm_raw)[i]}/{cm_raw.sum(1)[i]}" if i == j else str(int(val))
            )
            color = "white" if val > cm_raw.max() / 2 else "black"
            ax_raw.text(j, i, text, ha="center", va="center", color=color)
    ax_raw.set_xticks(range(len(names)))
    ax_raw.set_yticks(range(len(names)))
    ax_raw.set_xticklabels(names, rotation=45, ha="right")
    ax_raw.set_yticklabels(names)
    ax_raw.set_xlabel("Predicted", fontsize=11)
    ax_raw.set_ylabel("True", fontsize=11)
    ax_raw.set_title("Confusion Matrix (Raw)", fontsize=13)
    plt.colorbar(im_raw, ax=ax_raw, fraction=0.046, pad=0.04)
    fig_raw.tight_layout()

    # Normalized matrix plot
    fig_norm, ax_norm = plt.subplots(figsize=(6, 6))
    im_norm = ax_norm.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(len(names)):
        for j in range(len(names)):
            val = cm_norm[i, j]
            if val == 0:
                continue
            text = f"{val:.2f}"
            color = "white" if val > 0.5 else "black"
            ax_norm.text(j, i, text, ha="center", va="center", color=color)
    ax_norm.set_xticks(range(len(names)))
    ax_norm.set_yticks(range(len(names)))
    ax_norm.set_xticklabels(names, rotation=45, ha="right")
    ax_norm.set_yticklabels(names)
    ax_norm.set_xlabel("Predicted", fontsize=11)
    ax_norm.set_ylabel("True", fontsize=11)
    ax_norm.set_title("Confusion Matrix (Normalized)", fontsize=13)
    plt.colorbar(im_norm, ax=ax_norm, fraction=0.046, pad=0.04)
    fig_norm.tight_layout()

    print("[INFO] Confusion matrices plotted (raw and normalized).")
    return {"raw": fig_raw, "normalized": fig_norm}


def plot_child_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    figsize: Tuple[int, int] = (4, 4),
) -> Dict[str, plt.Figure]:
    """
    Plot the Adult (0) / Child (1) binary confusion matrix, returning both
    the raw counts and the row‑normalized version.

    Args:
        y_true (List[int]): Ground‑truth labels (0 = adult, 1 = child).
        y_pred (List[int]): Predicted labels  (0 = adult, 1 = child).
        figsize (Tuple[int, int]): Size of the output figures.

    Returns:
        Dict[str, plt.Figure]: A dict with keys **"raw"** and **"normalized"**
        mapping to the corresponding matplotlib figures.
    """
    # ------------------------------------------------------------------ #
    # 1) Compute raw and normalized matrices
    # ------------------------------------------------------------------ #
    cm_raw = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm_raw.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums != 0)

    classes = ["Adult", "Child"]

    # ------------------------------------------------------------------ #
    # 2) Plot raw confusion matrix
    # ------------------------------------------------------------------ #
    fig_raw, ax_raw = plt.subplots(figsize=figsize)
    im_raw = ax_raw.imshow(cm_raw, cmap="Blues")

    for i in range(2):
        for j in range(2):
            val = cm_raw[i, j]
            if val == 0:
                continue
            ax_raw.text(
                j,
                i,
                int(val),
                ha="center",
                va="center",
                color="white" if val > cm_raw.max() / 2 else "black",
            )

    ax_raw.set_xticks([0, 1])
    ax_raw.set_yticks([0, 1])
    ax_raw.set_xticklabels(classes)
    ax_raw.set_yticklabels(classes)
    ax_raw.set_xlabel("Predicted", fontsize=11)
    ax_raw.set_ylabel("True", fontsize=11)
    ax_raw.set_title("Adult / Child Confusion Matrix (Raw)", fontsize=13)
    plt.colorbar(im_raw, ax=ax_raw, fraction=0.046, pad=0.04)
    fig_raw.tight_layout()

    # ------------------------------------------------------------------ #
    # 3) Plot normalized confusion matrix
    # ------------------------------------------------------------------ #
    fig_norm, ax_norm = plt.subplots(figsize=figsize)
    im_norm = ax_norm.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)

    for i in range(2):
        for j in range(2):
            val = cm_norm[i, j]
            if val == 0:
                continue
            ax_norm.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if val > 0.5 else "black",
            )

    ax_norm.set_xticks([0, 1])
    ax_norm.set_yticks([0, 1])
    ax_norm.set_xticklabels(classes)
    ax_norm.set_yticklabels(classes)
    ax_norm.set_xlabel("Predicted", fontsize=11)
    ax_norm.set_ylabel("True", fontsize=11)
    ax_norm.set_title("Adult / Child Confusion Matrix (Normalized)", fontsize=13)
    plt.colorbar(im_norm, ax=ax_norm, fraction=0.046, pad=0.04)
    fig_norm.tight_layout()

    print("[INFO] Adult/Child confusion matrices plotted.")
    return {"raw": fig_raw, "normalized": fig_norm}


def plot_boxplots(
    data: List[Dict[str, Any]],
    x_field: str,
    y_field: str,
    title: str,
    labels_map: Dict[int, str],
    y_lim: Tuple[float, float] = None,
    cmap_name: str = "tab10",
) -> plt.Figure:
    """
    Draws class-wise colored boxplots for any metric (IoU, angle error, etc.)
    and includes a legend with mean ± std per class.


    Args:
        data (List[Dict[str, Any]]): List of metric dictionaries with 'class' and value fields.
        x_field (str): Name of the field to group by (class).
        y_field (str): Metric name to plot.
        title (str): Plot title.
        labels_map (Dict[int, str]): Mapping from class index to label.
        y_lim (Tuple[float, float], optional): Y-axis limits.
        cmap_name (str): Name of the colormap to use.

    Returns:
        matplotlib.figure.Figure: Boxplot figure.
    """
    classes = list(labels_map.keys())
    class_names = [labels_map[c] for c in classes]

    # Organize values by class
    values = [
        [d[y_field] for d in data if d[x_field] == labels_map[c]] for c in classes
    ]

    # Compute mean ± std for each class
    mean_std_text = {}
    for i, val in enumerate(values):
        name = class_names[i]
        if val:
            mu = np.mean(val)
            sigma = np.std(val)
            mean_std_text[name] = f"{mu:.2f} ± {sigma:.2f}"
        else:
            mean_std_text[name] = "N/A"

    fig, ax = plt.subplots(figsize=(9, 6))

    # Basic boxplot (unstyled)
    bp = ax.boxplot(
        values,
        positions=np.arange(len(class_names)),
        notch=True,
        patch_artist=True,
        boxprops=dict(facecolor="none", edgecolor="black"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    # Apply colormap
    cmap = plt.get_cmap(cmap_name)
    colors = {}
    for i, box in enumerate(bp["boxes"]):
        this_color = cmap(i)
        colors[class_names[i]] = this_color
        box.set_facecolor(this_color)
        box.set_edgecolor("black")

    # Add jittered points
    for i, (name, val) in enumerate(zip(class_names, values)):
        if val:
            jittered_x = np.random.normal(i, 0.04, size=len(val))
            ax.scatter(
                jittered_x,
                val,
                alpha=0.7,
                edgecolors="black",
                color=colors[name],
                label=f"{name} {mean_std_text[name]}",
            )

    # Axes style
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel(y_field, fontsize=12)
    ax.set_title(title, fontsize=14)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Legend outside
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.04, 1.0),
        frameon=False,
        title=f"{y_field} per class",
    )

    plt.tight_layout()
    print(f"[INFO] Boxplot for '{y_field}' created.")
    return fig


def plot_f1_vs_threshold(
    all_gts: List[int],
    all_scores: List[float],
    all_preds: List[int],
    labels_map: Dict[int, str],
    default_th: float = 0.5,
    n_steps: int = 100,
    sigma: float = 2.0,
) -> plt.Figure:
    """
    Plots F1 Score vs. confidence threshold for each class, reconstructing
    predictions from all_scores and all_preds at each threshold.

    Args:
        all_gts      : List of true labels (integers).
        all_scores   : List of scores (float) associated with each prediction.
        all_preds    : List of originally predicted labels (but the previous threshold will be ignored).
        labels_map   : Dict[int, str] mapping index→class name.
        default_th   : “default” threshold (used only to show it as a reference).
        n_steps      : Number of equally spaced points in [0,1] to evaluate F1.
        sigma        : Smoothing factor for the curve.

    Returns:
        matplotlib.figure.Figure with F1 vs threshold curves per class.
    """
    thresholds = np.linspace(0.0, 1.0, n_steps)
    y_true = np.array(all_gts, dtype=int)
    classes = list(labels_map.keys())

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("F1 vs. Confidence Threshold", fontsize=13)

    for cls in classes:
        f1s = []
        for t in thresholds:
            # For each prediction: if score >= t, assign the original label;
            # if score < t, assign -1 (background)
            y_pred_t = [
                (lbl if sc >= t else -1) for sc, lbl in zip(all_scores, all_preds)
            ]
            # Compute F1 with zero_division=0
            f1_val = f1_score(
                y_true, y_pred_t, labels=classes, average=None, zero_division=0
            )
            f1s.append(f1_val[classes.index(cls)])

        f1s = np.array(f1s)
        f1_s = smooth_curve(f1s, sigma)
        ax.plot(thresholds, f1_s, lw=2, label=f"{labels_map[cls]} {f1_s.mean():.3f}")

        # Mark the optimal F1 point for this class
        best_i = f1_s.argmax()
        ax.axvline(
            thresholds[best_i],
            linestyle="--",
            lw=1,
            color=ax.get_lines()[-1].get_color(),
        )
        ax.scatter(
            [thresholds[best_i]],
            [f1_s[best_i]],
            s=50,
            zorder=3,
            color=ax.get_lines()[-1].get_color(),
        )

    # Reference to the default threshold
    ax.axvline(
        default_th,
        color="gray",
        linestyle=":",
        linewidth=1.0,
        label=f"default_th={default_th}",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence Threshold", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1.0), fontsize=10, frameon=False)
    plt.tight_layout()
    print("[INFO] F1 vs. threshold curve plotted.")
    return fig


# -------------------------------------------------------------
# IV. Qualitative Grid & Saving Individually
# -----------------------------------------------------------------------------


def plot_qualitative_grid(
    samples: List[
        Tuple[
            Any, Dict[str, torch.Tensor], str, torch.Tensor, torch.Tensor, torch.Tensor
        ]
    ],
    labels_map: Dict[int, str],
    grid_shape: Tuple[int, int],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> plt.Figure:
    """
    Creates a qualitative grid of predictions and ground truths with OBBs.

    Args:
        samples (List): List of (image_tensor, prediction_dict, filename, gt_boxes, gt_angles, gt_labels).
        labels_map (Dict[int, str]): Mapping from class index to name.
        grid_shape (Tuple[int, int]): Rows and columns of the grid.
        mean (Tuple[float, float, float]): Mean values for image normalization.
        std (Tuple[float, float, float]): Std values for image normalization.

    Returns:
        matplotlib.figure.Figure: Qualitative grid figure.
    """
    rows, cols = grid_shape
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 4), facecolor="white"
    )
    axes = axes.flatten()

    for ax, (img_t, out, fname, gt_b, gt_a, gt_l, fp_img, fn_img, viz) in zip(
        axes, samples[: rows * cols]
    ):
        ax.imshow(denormalize_image(img_t, mean=mean, std=std))
        ax.axis("off")
        ax.set_title(f"{Path(fname).name}\nFP:{fp_img}  FN:{fn_img}", fontsize=7)
        ax.set_aspect("equal")

        # Ground Truth OBBs
        for pts, angle, cls in zip(gt_b, gt_a, gt_l):
            pts_np = pts.view(4, 2).numpy()
            ax.add_patch(
                patches.Polygon(
                    pts_np,
                    closed=True,
                    fill=False,
                    edgecolor="#008000",
                    linewidth=2,
                    linestyle="--",
                )
            )
            ax.plot(pts_np[[0, 1], 0], pts_np[[0, 1], 1], color="orange", linewidth=2)
            br_x, br_y = pts_np[:, 0].max(), pts_np[:, 1].max()
            ax.text(
                br_x,
                br_y,
                f"{labels_map.get(int(cls), 'unknown')}: {math.degrees(float(angle)):.1f}°",
                color="white",
                fontsize=6,
                fontweight="bold",
                ha="right",
                va="bottom",
                bbox=dict(facecolor="#008000", alpha=0.8, edgecolor="none", pad=2.5),
            )

        # Predicted OBBs
        for i, (pts, lbl, score) in enumerate(
            zip(out["polygons"], out["labels"], out["scores"])
        ):
            pts_np = pts.view(4, 2).numpy()
            ax.add_patch(
                patches.Polygon(
                    pts_np,
                    closed=True,
                    fill=False,
                    edgecolor="#004080",
                    linewidth=1.5,
                )
            )
            ax.plot(
                pts_np[[0, 1], 0], pts_np[[0, 1], 1], color="#800000", linewidth=1.5
            )
            tl_x, tl_y = pts_np[:, 0].min(), pts_np[:, 1].min()
            ang = math.degrees(float(out["boxes"][i, 4]))
            ax.text(
                tl_x,
                tl_y,
                f"{labels_map.get(int(lbl), 'unknown')}: {ang:.1f}° / {score:.2f}",
                color="white",
                fontsize=6,
                ha="left",
                va="top",
                bbox=dict(facecolor="#004080", alpha=0.9, edgecolor="none", pad=2.5),
            )

    # Hide any unused axes
    for ax in axes[len(samples) :]:
        ax.axis("off")

    fig.tight_layout(pad=0.5)
    print("[INFO] Grid of qualitative predictions plotted.")
    return fig


def save_individual_predictions(
    samples: List[
        Tuple[
            Any,
            Dict[str, torch.Tensor],
            str,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
            int,
            Optional[Dict[str, Any]],
        ]
    ],
    labels_map: Dict[int, str],
    output_dir: str,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    split_by_error: bool = True,
    viz_original_res: bool = False,
    orig_size_resolver: Optional[Callable[[str], Optional[Tuple[int, int]]]] = None,
    resize_size: Tuple[int, int] = (640, 640),
) -> None:
    """
    Saves individual visualizations of predictions with ground truth (GT) and predicted oriented bounding boxes (OBBs).

    This function generates and saves images showing both GT and predicted OBBs for qualitative analysis.
    The images can be optionally split into subdirectories based on error types (e.g., false positives, false negatives).

    Args:
        samples (List): List of tuples containing:
            - Image tensor (torch.Tensor)
            - Prediction dictionary (Dict[str, torch.Tensor])
            - File name (str)
            - Ground truth boxes (torch.Tensor)
            - Ground truth angles (torch.Tensor)
            - Ground truth labels (torch.Tensor)
            - False positive count (int)
            - False negative count (int)
        labels_map (Dict[int, str]): Mapping from class indices to human-readable labels.
        output_dir (str): Directory where the visualizations will be saved.
        mean (Tuple[float, float, float]): Mean values for image normalization (used for denormalization).
        std (Tuple[float, float, float]): Standard deviation values for image normalization (used for denormalization).
        split_by_error (bool): Whether to split saved images into subdirectories based on error types.

    Returns:
        None: Saves visualizations to the specified output directory.
    """
    # Convert output directory to Path object
    base_dir = Path(output_dir)

    # Create subdirectories for different error types if splitting is enabled
    if split_by_error:
        for sub in ("tp_only", "fp", "fn", "fp_fn"):
            (base_dir / sub).mkdir(parents=True, exist_ok=True)
    else:
        base_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over each sample in the dataset
    for sample in samples:
        # Desempaquetado compatible (con o sin viz_payload)
        if len(sample) == 9:
            img_t, out, fname, gt_b, gt_a, gt_l, fp_img, fn_img, viz = sample
        else:
            img_t, out, fname, gt_b, gt_a, gt_l, fp_img, fn_img = sample
            viz = None

        fig, ax = plt.subplots(figsize=(6, 6))

        # Fondo y factores de escala
        # ------------------------------------------------------------
        # 1) Preparar fondo (base_img) y factores de escala (sx, sy)
        # ------------------------------------------------------------
        if viz is not None and viz.get("orig_img", None) is not None:
            base_img = viz["orig_img"]                      # np.uint8 (H0, W0, 3)
            sx, sy = viz["scale"]
        else:
            base_img = denormalize_image(img_t, mean=mean, std=std)  # (Hr, Wr, 3)
            sx, sy = 1.0, 1.0
            if viz_original_res and orig_size_resolver is not None:
                wh = orig_size_resolver(fname)
                if wh is not None:
                    W0, H0 = wh
                    Wr, Hr = resize_size
                    sx = float(W0) / float(Wr)
                    sy = float(H0) / float(Hr)
                    try:
                        from PIL import Image
                        base_img = np.asarray(Image.fromarray(base_img).resize((W0, H0)))
                    except Exception:
                        sx, sy = 1.0, 1.0  # fallback a 640 si algo falla

        H_out, W_out = int(base_img.shape[0]), int(base_img.shape[1])

        # ------------------------------------------------------------
        # 2) Crear figura EXACTAMENTE del tamaño del fondo
        #    (W_out x H_out px)  => figsize = (W_out/dpi, H_out/dpi)
        # ------------------------------------------------------------
        dpi = 100
        fig = plt.figure(figsize=(W_out / dpi, H_out / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  # ocupar todo el lienzo sin bordes
        # Mostrar la imagen SIN márgenes/antialias en los bordes:
        ax.imshow(base_img, extent=(0, W_out, H_out, 0), interpolation="nearest")
        ax.set_xlim(0, W_out)
        ax.set_ylim(H_out, 0)  # invertir eje Y para coords tipo imagen
        ax.axis("off")
        # Draw ground truth OBBs
        for pts, ang, lbl in zip(gt_b, gt_a, gt_l):
            coords = pts.detach().cpu().view(4, 2).numpy()
            # Escalar a resolución original si corresponde
            coords[:, 0] *= sx
            coords[:, 1] *= sy
            ax.add_patch(
                patches.Polygon(
                    coords,
                    closed=True,
                    fill=False,
                    edgecolor="#008000",  # Dark green for GT
                    linewidth=2,
                    linestyle="--",
                )
            )
            ax.plot(coords[[0, 1], 0], coords[[0, 1], 1], color="orange", linewidth=2)

            # Add label and angle text at the bottom-right corner of the OBB
            br_x, br_y = coords[:, 0].max(), coords[:, 1].max()
            ax.text(
                br_x,
                br_y,
                f"{labels_map.get(int(lbl), 'unknown')}: {math.degrees(float(ang)):.1f}°",
                color="white",
                fontsize=6,
                fontweight="bold",
                ha="right",
                va="bottom",
                bbox=dict(facecolor="#008000", alpha=0.8, edgecolor="none", pad=2.5),
            )

        # Draw predicted OBBs
        for i, (pts, lbl, score) in enumerate(
            zip(out["polygons"], out["labels"], out["scores"])
        ):
            coords = pts.cpu().view(4, 2).numpy()
            coords[:, 0] *= sx
            coords[:, 1] *= sy  # Convert tensor to numpy array
            ax.add_patch(
                patches.Polygon(
                    coords,
                    closed=True,
                    fill=False,
                    edgecolor="#004080",  # Dark blue for predictions
                    linewidth=1.5,
                )
            )
            ax.plot(
                coords[[0, 1], 0], coords[[0, 1], 1], color="#800000", linewidth=1.5
            )

            # Add label, angle, and confidence score text at the top-left corner of the OBB
            tl_x, tl_y = coords[:, 0].min(), coords[:, 1].min()
            ang_pred = math.degrees(float(out["boxes"][i, 4]))
            ax.text(
                tl_x,
                tl_y,
                f"{labels_map.get(int(lbl), 'unknown')}: {ang_pred:.1f}° / {score:.2f}",
                color="white",
                fontsize=6,
                ha="left",
                va="top",
                bbox=dict(facecolor="#004080", alpha=0.9, edgecolor="none", pad=2.5),
            )

        # Determine the subdirectory based on error type if splitting is enabled
        if not split_by_error:
            save_dir = base_dir
        else:
            if fp_img and not fn_img:
                subdir = "fp"  # False positives only
            elif fn_img and not fp_img:
                subdir = "fn"  # False negatives only
            elif fp_img and fn_img:
                subdir = "fp_fn"  # Both false positives and false negatives
            else:
                subdir = "tp_only"  # True positives only

            save_dir = base_dir / subdir
            save_dir.mkdir(exist_ok=True, parents=True)

        # Save the visualization to the appropriate directory
        fig.savefig(save_dir / Path(fname).name, dpi=dpi, bbox_inches=None, pad_inches=0)
        plt.close(fig)  # Close the figure to free memory

    print(f"[INFO] Saved individual predictions to {output_dir}")


# -----------------------------------------------------------------------------
# V. Main Entry
# -----------------------------------------------------------------------------


def inference(
    model: torch.nn.Module,
    test_loader: DataLoader,
    output_dir: Union[str, Path],
    device: torch.device,
    labels_map: Dict[int, str],
    scale_factors: List[float],
    ratio_factors: List[float],
    face_thres: float = 0.25,
    baby_thres: float = 0.25,
    iou_thres: float = 0.5,
    class_thres: float = 0.5,
    grid_shape: Tuple[int, int] = (3, 3),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    save_figs: bool = True,
    close_figs: bool = True,
    anchors_cache_path: Union[str, Path] = None,
    render_original: bool = False,
) -> Dict[str, Any]:
    """
        This function processes a test dataset through a trained model and generates comprehensive
    evaluation metrics and visualizations.

    Steps:
        1. Anchor preparation for inference
        2. Model inference on test set
        3. Metrics computation and visualization generation
        4. CSV export of metrics and confusion matrices
        5. Saving of prediction visualizations

        - model (torch.nn.Module): Trained model for inference.
        - test_loader (DataLoader): DataLoader containing test dataset.
        - output_dir (Union[str, Path]): Directory path to save results and visualizations.
        - device (torch.device): Computing device ('cuda' or 'cpu').
        - labels_map (Dict[int, str]): Mapping of class indices to label names.
        - scale_factors (List[float]): Scale factors for anchor box generation.
        - ratio_factors (List[float]): Aspect ratio factors for anchor box generation.
        - face_thres (float, optional): Confidence threshold for face detection. Defaults to 0.25.
        - baby_thres (float, optional): Confidence threshold for baby classification. Defaults to 0.25.
        - iou_thres (float, optional): IoU threshold for prediction matching. Defaults to 0.5.
        - class_thres (float, optional): Confidence threshold for class predictions. Defaults to 0.5.
        - grid_shape (Tuple[int, int], optional): Shape of prediction visualization grid (rows, cols).
            Defaults to (3, 3).
        - mean (Tuple[float, float, float], optional): Mean values for image normalization.
            Defaults to (0.485, 0.456, 0.406).
        - std (Tuple[float, float, float], optional): Standard deviation values for image normalization.
            Defaults to (0.229, 0.224, 0.225).
        - save_figs (bool, optional): Whether to save generated figures. Defaults to True.
        - close_figs (bool, optional): Whether to close figures after saving. Defaults to True.
        - anchors_cache_path (Union[str, Path], optional): Path to cache generated anchors.
            Defaults to None.

            - "mAP": Mean Average Precision across all classes
            - "APs": Dictionary of per-class Average Precision scores
        -  render_original (bool, optional): Whether to render predictions on original images. Defaults to False.

    Generated Outputs:
        - Precision-Recall curves
        - Confusion matrices (raw and normalized) for class predictions
        - Confusion matrices (raw and normalized) for child/adult classification
        - IoU distribution boxplots per class
        - Angle error distribution boxplots per class
        - F1 score vs confidence threshold plots
        - Grid of qualitative prediction examples
        - Individual prediction visualizations
        - CSV files with metrics and confusion matrices
    """

    def save_figure(fig: plt.Figure, fname: str):
        """Helper to save a matplotlib figure if enabled."""
        if save_figs:
            fig.savefig(figures_dir / fname, dpi=150, bbox_inches="tight")
            if close_figs:
                plt.close(fig)

    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    predictions_dir = output_dir / "predictions"

    # Create output directories if they do not exist
    for d in (figures_dir, predictions_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("[STEP 1] Preparing anchors...")
    resize_size, anchors_xy, _ = prepare_anchors(
        model=model,
        loader=test_loader,
        device=device,
        scale_factors=scale_factors,
        ratio_factors=ratio_factors,
        anchors_cache_path=anchors_cache_path,
    )

    print("[STEP 2] Running inference...")
    results = run_inference(
        model=model,
        loader=test_loader,
        anchors_xy=anchors_xy,
        resize_size=resize_size,
        face_thres=face_thres,
        iou_thres=iou_thres,
        class_thres=class_thres,
        baby_thres=baby_thres,
        device=device,
        labels_map=labels_map,
        render_original=render_original,
    )

    print("[STEP 3] Computing metrics and plots...")
    mAP, APs = compute_map_and_pr(results["per_true"], results["per_score"])

    # Precision-Recall curve
    save_figure(
        plot_precision_recall(
            results["per_true"], results["per_score"], labels_map, mAP
        ),
        "precision_recall.png",
    )

    # Confusion matrices (raw and normalized)
    cm_figs = plot_child_confusion_matrix(
        y_true=results["child_gt"],
        y_pred=results["child_pred"],
    )
    save_figure(cm_figs["raw"], "child_cm_raw.png")
    save_figure(cm_figs["normalized"], "child_cm_normalized.png")

    # Confusion matrices (raw and normalized)
    cm_figs = plot_confusion_matrix(
        y_true=results["y_true"], y_pred=results["y_pred"], labels_map=labels_map
    )
    save_figure(cm_figs["raw"], "class_cm_raw.png")
    save_figure(cm_figs["normalized"], "class_cm_normalized.png")

    # IoU boxplots per class
    iou_data = [
        {"class": labels_map[c], "iou": v}
        for c, vals in results["iou_errs"].items()
        for v in vals
    ]
    save_figure(
        plot_boxplots(
            iou_data,
            "class",
            "iou",
            "IoU Distribution per Class",
            labels_map,
            y_lim=(0, 1),
        ),
        "iou_boxplot.png",
    )

    # Angle error boxplots per class
    ang_data = [
        {"class": labels_map[c], "error°": v}
        for c, vals in results["angle_errs"].items()
        for v in vals
    ]
    save_figure(
        plot_boxplots(
            ang_data,
            "class",
            "error°",
            "Angle-Error Distribution per Class",
            labels_map,
            y_lim=(0, 180),
        ),
        "angle_boxplot.png",
    )

    # F1 score vs. confidence threshold
    save_figure(
        plot_f1_vs_threshold(
            results["all_gts"], results["all_scores"], results["all_preds"], labels_map
        ),
        "f1_threshold.png",
    )

    # Qualitative grid of predictions
    save_figure(
        plot_qualitative_grid(results["samples"], labels_map, grid_shape, mean, std),
        "grid_examples.png",
    )

    print("[STEP 4] Exporting metrics and confusion matrix CSV...")
    metrics_csv = export_metrics_and_confusion_csv(results, labels_map, output_dir)

    print("[STEP 5] Saving individual prediction images...")
    resolver = _build_image_size_resolver(test_loader.dataset)
    save_individual_predictions(
        samples=results["samples"],
        labels_map=labels_map,
        output_dir=predictions_dir,
        mean=mean,
        std=std,
        split_by_error=True,
        viz_original_res=render_original,
        orig_size_resolver=resolver,
        resize_size=resize_size,
    )
    print("[DONE] Inference and reporting completed.")

    return {"mAP": mAP, "APs": APs}


def plot_training_curves_from_csv(csv_path: str, output_dir: Path) -> None:
    """
    Creates and saves training/validation curves from a model training log CSV.

    Generates separate plots for different loss components and metrics:
    - Face detection loss
    - Classification loss
    - Angular regression loss
    - Oriented bounding box (OBB) loss
    - Total combined loss
    - Mean Average Precision (mAP)

    Each plot shows training and validation curves (except mAP which is validation only)
    with epochs on x-axis and the corresponding metric on y-axis.

    Args:
        csv_path (str): Path to CSV file containing training logs with columns:
            epoch, train_face_loss, test_face_loss, train_class_loss, etc.
        output_dir (Path): Directory where plots will be saved under 'curves' subfolder.
            Will be created if it doesn't exist.
    """
    # Read training log data
    df = pd.read_csv(csv_path)
    curves_path = output_dir / "curves"
    curves_path.mkdir(parents=True, exist_ok=True)

    def make_plot(train_col: str, val_col: str, title: str, ylabel: str, filename: str):
        """Helper function to create and save a single training/validation curve plot"""
        plt.figure(figsize=(10, 6), facecolor="white")

        # Plot training curve
        plt.plot(
            df["epoch"],
            df[train_col],
            label="Training",
            color="#2ecc71",  # Bright green
            linewidth=2,
            marker="o",
            markersize=6,
            alpha=0.8,
        )

        # Plot validation curve
        plt.plot(
            df["epoch"],
            df[val_col],
            label="Validation",
            color="#e74c3c",  # Bright red
            linewidth=2,
            marker="s",
            markersize=6,
            alpha=0.8,
        )

        # Styling
        plt.xlabel("Epoch", fontsize=12, labelpad=10)
        plt.ylabel(ylabel, fontsize=12, labelpad=10)
        plt.title(title, fontsize=14, pad=15)

        # Grid and background
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.gca().set_facecolor("#f8f9fa")  # Light gray background

        # Legend with semi-transparent background
        plt.legend(
            framealpha=0.95,
            facecolor="white",
            edgecolor="none",
            fontsize=10,
            loc="upper right",
        )

        plt.tight_layout()
        plt.savefig(curves_path / f"{filename}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Generate individual loss component plots
    make_plot(
        "train_face_loss",
        "test_face_loss",
        "Face Detection Loss Over Training",
        "Loss",
        "face_curves",
    )
    make_plot(
        "train_class_loss",
        "test_class_loss",
        "Classification Loss Over Training",
        "Loss",
        "class_curves",
    )
    make_plot(
        "train_angular_loss",
        "test_angular_loss",
        "Angular Regression Loss Over Training",
        "Loss",
        "angle_curves",
    )
    make_plot(
        "train_obb_loss",
        "test_obb_loss",
        "OBB Regression Loss Over Training",
        "Loss",
        "obb_curves",
    )
    make_plot(
        "train_rect_loss",
        "test_rect_loss",
        "Orthogonality Regularization Over Training",
        "Loss",
        "regularization_curves",
    )
    make_plot(
        "train_child_loss",
        "test_child_loss",
        "Child Loss Over Training",
        "Loss",
        "child_curves",
    )
    make_plot(
        "train_total_loss",
        "test_total_loss",
        "Total Combined Loss Over Training",
        "Loss",
        "total_curves",
    )

    # mAP plot (validation only)
    plt.figure(figsize=(10, 6), facecolor="white")
    plt.plot(
        df["epoch"],
        df["test_mAP"],
        label="Validation mAP",
        color="#3498db",  # Bright blue
        linewidth=2.5,
        marker="D",
        markersize=7,
    )

    plt.xlabel("Epoch", fontsize=12, labelpad=10)
    plt.ylabel("mAP", fontsize=12, labelpad=10)
    plt.title("Mean Average Precision Over Training", fontsize=14, pad=15)

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.gca().set_facecolor("#f8f9fa")
    plt.legend(framealpha=0.95, facecolor="white", edgecolor="none", fontsize=10)

    plt.tight_layout()
    plt.savefig(curves_path / "map.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Training curves saved to: {curves_path}")


def export_metrics_and_confusion_csv(
    results: dict, labels_map: dict[int, str], out_dir: Path, fname: str = "metrics.csv"
) -> Path:
    """
    Exports evaluation results to a single CSV file containing:
      - Per-class metrics table (TP, FP, FN, Precision, Recall, F1, AP, IoU, Angle error)
      - Raw confusion matrix
      - Normalized confusion matrix

    Each section is separated by a comment line starting with '# --- SECTION ---'.
    This allows pandas.read_csv(..., comment='#') to read each table independently.

    Args:
        results (dict): Output dictionary from the inference pipeline containing metrics and predictions.
        labels_map (dict[int, str]): Mapping from class indices to human-readable class names.
        out_dir (Path): Directory where the CSV will be saved.
        fname (str): Name of the CSV file (default: "metrics.csv").

    Returns:
        Path: Path to the saved CSV file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / fname

    classes = list(labels_map.keys())
    class_names = [labels_map[c] for c in classes]
    bg_label, bg_name = -1, "BG"

    # ---------- Per-class metrics table ----------------------------------------
    rows = []
    for c, name in zip(classes, class_names):
        tp = results["stats"][c]["tp"]
        fp = results["stats"][c]["fp"]
        fn = results["stats"][c]["fn"]

        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        ap = (
            average_precision_score(results["per_true"][c], results["per_score"][c])
            if tp
            else 0.0
        )

        iou_vals = results["iou_errs"][c]
        angle_vals = results["angle_errs"][c]

        rows.append(
            dict(
                Class=name,
                TP=tp,
                FP=fp,
                FN=fn,
                Precision=prec,
                Recall=rec,
                F1=f1,
                AP_PR=ap,
                IoU_mean=np.mean(iou_vals) if iou_vals else 0.0,
                IoU_std=np.std(iou_vals) if iou_vals else 0.0,
                Angle_mean_deg=np.mean(angle_vals) if angle_vals else 0.0,
                Angle_std_deg=np.std(angle_vals) if angle_vals else 0.0,
            )
        )

    # Add background row: counts FPs where prediction was made but no GT exists
    bg_fp = int((np.array(results["y_true"]) == bg_label).sum())
    rows.append(
        dict(
            Class=bg_name,
            TP=0,
            FP=bg_fp,
            FN=0,
            Precision=0.0,
            Recall=0.0,
            F1=0.0,
            AP_PR=0.0,
            IoU_mean=0.0,
            IoU_std=0.0,
            Angle_mean_deg=0.0,
            Angle_std_deg=0.0,
        )
    )

    df_metrics = pd.DataFrame(rows).set_index("Class")

    # ---------- Confusion matrices (raw and normalized) ------------------------
    mat_labels = classes + [bg_label]
    mat_names = class_names + [bg_name]

    cm_raw = confusion_matrix(results["y_true"], results["y_pred"], labels=mat_labels)
    cm_norm = np.nan_to_num(cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True))

    df_cm_raw = pd.DataFrame(cm_raw, index=mat_names, columns=mat_names)
    df_cm_norm = pd.DataFrame(cm_norm, index=mat_names, columns=mat_names)

    # ---------- Write all tables to a single CSV file --------------------------
    with open(csv_path, "w", newline="") as f:
        f.write(
            "# --- METRICS PER CLASS -------------------------------------------------\n"
        )
        df_metrics.to_csv(f, float_format="%.4f")
        f.write(
            "\n# --- CONFUSION MATRIX RAW ----------------------------------------------\n"
        )
        df_cm_raw.to_csv(f)
        f.write(
            "\n# --- CONFUSION MATRIX NORMALIZED ---------------------------------------\n"
        )
        df_cm_norm.to_csv(f, float_format="%.4f")

    print(f"[INFO] Metrics and confusion matrices saved to {csv_path}")
    return csv_path


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def _build_image_size_resolver(dataset, images_subdir: str = "images"):
    """
    Devuelve una función que, dado fname (stem o ruta), retorna (W0, H0) o None.
    """
    root = Path(dataset.root_dir) / dataset.split / images_subdir
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def _resolve(fname: str):
        p = Path(fname)
        # 1) Si es ruta directa
        if p.is_file():
            with Image.open(p) as im:
                return im.size  # (W,H)
        # 2) Si es solo el stem, probar en root
        stem = p.stem if p.suffix else p.name
        for e in exts:
            cand = root / f"{stem}{e}"
            if cand.exists():
                with Image.open(cand) as im:
                    return im.size  # (W,H)
        return None

    return _resolve


def _load_original_and_scale(dataset, fname: str, resize_size: Tuple[int, int]):
    """
    Carga la imagen original y calcula escala (sx, sy) relativa a resize_size (W_r, H_r).
    Retorna: (np_img_rgb, (sx, sy)). Si no se puede, vuelve a None y (1,1).
    """
    resolver = _build_image_size_resolver(dataset)
    wh = resolver(fname)
    if wh is None:
        return None, (1.0, 1.0)
    W0, H0 = wh  # PIL (W,H)
    Wr, Hr = resize_size
    sx = float(W0) / float(Wr)
    sy = float(H0) / float(Hr)
    # cargar imagen original real como fondo
    root = Path(dataset.root_dir) / dataset.split / "images"
    p = Path(fname)
    if not p.is_file():
        stem = p.stem if p.suffix else p.name
        for e in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            cand = root / f"{stem}{e}"
            if cand.exists():
                p = cand
                break
    try:
        with Image.open(p) as im:
            im = im.convert("RGB")
            np_img = np.asarray(im)
            return np_img, (sx, sy)
    except Exception:
        return None, (1.0, 1.0)