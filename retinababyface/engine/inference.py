import os
import math
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union

import torch
import pandas as pd
import numpy as np
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
) -> Tuple[Tuple[int, int], torch.Tensor, torch.Tensor]:
    """
    Prepares anchors for inference using the base OBB statistics and given scale/ratio factors.

    Args:
        model (torch.nn.Module): The model used for feature extraction to generate anchors.
        loader (DataLoader): DataLoader to estimate image resize size.
        device (torch.device): Device for tensor creation (usually 'cuda' or 'cpu').
        scale_factors (List[float]): List of scaling factors for anchor generation.
        ratio_factors (List[float]): List of aspect ratio factors for anchors.
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
    conf_thres: float,
    iou_thres: float,
    class_thres: float,
    device: torch.device,
    labels_map: Dict[int, str],
) -> Dict[str, Any]:
    """
    Runs inference on a dataset and collects predictions, ground truths, and metrics.

    Args:
        model (torch.nn.Module): The trained model to use for inference.
        loader (DataLoader): DataLoader providing the test dataset.
        anchors_xy (torch.Tensor): Precomputed anchors in (x, y) format.
        resize_size (Tuple[int, int]): Target size used to resize images.
        conf_thres (float): Confidence threshold for filtering predictions.
        iou_thres (float): IoU threshold for matching predictions to ground truths.
        class_thres (float): Class score threshold for filtering predictions.
        device (torch.device): Device to run inference on (e.g., 'cuda' or 'cpu').
        labels_map (Dict[int, str]): Mapping from class indices to human-readable labels.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - per_true: Binary ground truth (1 for TP, 0 for FP) per class.
            - per_score: Confidence scores of predictions per class.
            - iou_errs: IoU errors per class.
            - angle_errs: Angle errors per class.
            - stats: TP, FP, FN counts per class.
            - y_true: Ground truth labels for confusion matrix.
            - y_pred: Predicted labels for confusion matrix.
            - all_gts: Ground truth labels for F1 vs. threshold.
            - all_preds: Predicted labels for F1 vs. threshold.
            - all_scores: Prediction scores for F1 vs. threshold.
            - samples: List of qualitative samples for visualization.
    """
    # Initialize data structures for metrics and qualitative results
    per_true = {c: [] for c in labels_map}
    per_score = {c: [] for c in labels_map}
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in labels_map}
    y_true, y_pred = [], []
    iou_errs = {c: [] for c in labels_map}
    angle_errs = {c: [] for c in labels_map}
    samples = []

    # Additional lists for F1 vs. threshold computation
    all_gts = []  # Ground truth labels
    all_preds = []  # Predicted labels
    all_scores = []  # Prediction scores

    dataset = loader.dataset
    global_idx = 0

    model.eval()  # Set model to evaluation mode
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference"):
            imgs = batch["image"].to(device)
            targets = batch["target"]

            # Perform inference with rotated NMS
            outputs = infer_with_rotated_nms(
                model, imgs, anchors_xy, resize_size, conf_thres, iou_thres, class_thres
            )
            orient_logits, _, _, _ = model(imgs)
            orientation_probs = F.softmax(orient_logits, dim=-1)

            batch_size = imgs.size(0)
            for b in range(batch_size):
                fname = dataset.file_list[global_idx]
                global_idx += 1

                # --------------------- Ground Truth Processing -------------------
                valid_mask = targets["valid_mask"][b]
                gt_boxes = targets["boxes"][b][valid_mask]  # GT boxes coordinates
                gt_angles = targets["angles"][b][valid_mask].view(
                    -1
                )  # GT rotation angles
                gt_labels = targets["class_idx"][b][valid_mask]  # GT class labels
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
                        y_true.append(-1)  # Background row
                        y_pred.append(cls_det)  # Predicted class column

                    # Store qualitative sample
                    samples.append(
                        (
                            imgs[b].cpu(),
                            {k: v.cpu().detach() for k, v in outputs[b].items()},
                            fname,
                            gt_boxes.cpu(),
                            gt_angles.cpu(),
                            gt_labels.cpu(),
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

                        if cls_det == true_cls:
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
                            stats[cls_det]["fp"] += 1
                            per_true[cls_det].append(0)
                            per_score[cls_det].append(score_det)

                            stats[true_cls]["fn"] += 1
                            gt_matched[best_gt_idx] = True

                            y_true.append(true_cls)  # GT class row
                            y_pred.append(cls_det)  # Predicted class column

                            all_gts.append(true_cls)
                            all_preds.append(cls_det)
                            all_scores.append(score_det)
                    else:
                        # ---------------- BACKGROUND FALSE POSITIVE --------------
                        # No matching GT with sufficient IoU
                        stats[cls_det]["fp"] += 1
                        per_true[cls_det].append(0)
                        per_score[cls_det].append(score_det)

                        y_true.append(-1)  # Background row
                        y_pred.append(cls_det)  # Predicted class column

                        all_gts.append(-1)
                        all_preds.append(true_cls)
                        all_scores.append(score_det)

                # ---- Process unmatched GT boxes as False Negatives -------------
                for i in range(num_gt):
                    if not gt_matched[i]:
                        cls_gt = int(gt_labels[i])
                        stats[cls_gt]["fn"] += 1
                        y_true.append(cls_gt)  # GT class row
                        y_pred.append(-1)  # Background column

                # ---- Store qualitative sample ----------------------------------
                samples.append(
                    (
                        imgs[b].cpu(),
                        {k: v.cpu().detach() for k, v in outputs[b].items()},
                        fname,
                        gt_boxes.cpu(),
                        gt_angles.cpu(),
                        gt_labels.cpu(),
                    )
                )

            # Clean up to free memory
            del imgs, outputs, targets
            torch.cuda.empty_cache()

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
) -> plt.Figure:
    """
    Plots a confusion matrix with absolute values and correct/total for diagonals.

    Args:
        y_true (List[int]): Ground truth class indices.
        y_pred (List[int]): Predicted class indices (may include -1 for background).
        labels_map (Dict[int, str]): Mapping from class index to class name.

    Returns:
        matplotlib.figure.Figure: Confusion matrix plot.
    """
    labels = list(labels_map.keys()) + [-1]
    names = [labels_map.get(l, "BG") for l in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0)

    # Annotate matrix with values
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j] == 0:
                continue
            text = f"{np.diag(cm)[i]}/{cm.sum(1)[i]}" if i == j else str(cm[i, j])
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color)

    # Axis and styling
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    print("[INFO] Confusion matrix plotted.")
    return fig


def plot_boxplots(
    data: List[Dict[str, Any]],
    x_field: str,
    y_field: str,
    title: str,
    labels_map: Dict[int, str],
    y_lim: Tuple[float, float] = None,
    cmap_name: str = "tab20",
) -> plt.Figure:
    """
    Draws class-wise colored boxplots for any metric (IoU, angle error, etc.).

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

    fig, ax = plt.subplots(figsize=(6, 4))

    # Basic boxplot (unstyled)
    bp = ax.boxplot(
        values,
        labels=class_names,
        notch=True,
        patch_artist=True,
        boxprops=dict(facecolor="none", edgecolor="black"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    # Apply colormap
    cmap = plt.get_cmap(cmap_name)
    for i, box in enumerate(bp["boxes"]):
        color = cmap(i)
        box.set_facecolor(color)
        box.set_edgecolor("black")

    # Add jittered points
    for i, vals in enumerate(values):
        xs = np.random.normal(i + 1, 0.06, size=len(vals))
        ax.scatter(xs, vals, color=cmap(i), s=6, alpha=0.7)

    # Axis and style
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(y_field, fontsize=11)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

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

    for ax, (img_t, out, fname, gt_b, gt_a, gt_l) in zip(axes, samples[: rows * cols]):
        ax.imshow(denormalize_image(img_t, mean=mean, std=std))
        ax.axis("off")
        ax.set_title(Path(fname).name, fontsize=8)

        # Draw GT polygons
        for pts, angle, cls in zip(gt_b, gt_a, gt_l):
            pts_np = pts.view(4, 2).numpy()
            ax.add_patch(
                patches.Polygon(
                    pts_np,
                    closed=True,
                    fill=False,
                    edgecolor="green",
                    linewidth=2,
                    linestyle="--",
                )
            )
            ax.plot(pts_np[[0, 1], 0], pts_np[[0, 1], 1], color="orange", linewidth=2)
            cen = pts_np.mean(axis=0)
            ax.text(
                cen[0],
                cen[1],
                f"{labels_map[int(cls)]}: {math.degrees(angle):.1f}°",
                color="green",
                fontsize=6,
                fontweight="bold",
                ha="center",
                va="center",
                path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
            )

        # Draw predicted polygons
        for i, (pts, lbl, score) in enumerate(
            zip(out["polygons"], out["labels"], out["scores"])
        ):
            pts_np = pts.view(4, 2).numpy()
            ax.add_patch(
                patches.Polygon(
                    pts_np,
                    closed=True,
                    fill=False,
                    edgecolor="blue",
                    linewidth=1.5,
                )
            )
            ax.plot(pts_np[[0, 1], 0], pts_np[[0, 1], 1], color="red", linewidth=1.5)
            cen = pts_np.mean(axis=0)
            ang = math.degrees(float(out["boxes"][i, 4]))
            ax.text(
                cen[0],
                cen[1],
                f"{labels_map[int(lbl)]}: {ang:.1f}°/{score:.2f}",
                color="blue",
                fontsize=5,
                ha="center",
                va="center",
                path_effects=[patheffects.withStroke(linewidth=2, foreground="black")],
            )

    # Hide unused axes
    for ax in axes[len(samples) :]:
        ax.axis("off")

    fig.tight_layout(pad=0.5)
    print("[INFO] Grid of qualitative predictions plotted.")
    return fig


def save_individual_predictions(
    samples: List[
        Tuple[
            Any, Dict[str, torch.Tensor], str, torch.Tensor, torch.Tensor, torch.Tensor
        ]
    ],
    labels_map: Dict[int, str],
    output_dir: str,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> None:
    """
    Saves individual visualizations of predictions with GT and predicted OBBs.

    Args:
        samples (List): Sample tuples with image tensor, predictions, file name, GT data.
        labels_map (Dict[int, str]): Mapping from class index to readable label.
        output_dir (str): Output directory for saving images.
        mean (Tuple): Mean for denormalization.
        std (Tuple): Std for denormalization.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for img_t, out, fname, gt_b, gt_a, gt_l in samples:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(denormalize_image(img_t, mean=mean, std=std))
        ax.axis("off")
        ax.set_aspect("equal")

        for pts, ang, lbl in zip(gt_b, gt_a, gt_l):
            coords = pts.view(4, 2).numpy()
            ax.add_patch(
                patches.Polygon(
                    coords,
                    closed=True,
                    fill=False,
                    edgecolor="green",
                    linewidth=2,
                    linestyle="--",
                )
            )
            ax.plot(coords[[0, 1], 0], coords[[0, 1], 1], color="orange", linewidth=2)
            ax.text(
                coords[:, 0].mean(),
                coords[:, 1].mean(),
                f"{labels_map[int(lbl)]}: {math.degrees(float(ang)):.1f}°",
                color="green",
                fontsize=6,
                fontweight="bold",
                ha="center",
                va="center",
                path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
            )

        for i, (pts, lbl, score) in enumerate(
            zip(out["polygons"], out["labels"], out["scores"])
        ):
            coords = pts.cpu().view(4, 2).numpy()
            ax.add_patch(
                patches.Polygon(
                    coords,
                    closed=True,
                    fill=False,
                    edgecolor="blue",
                    linewidth=1.5,
                )
            )
            ax.plot(coords[[0, 1], 0], coords[[0, 1], 1], color="red", linewidth=1.5)
            ang_pred = math.degrees(float(out["boxes"][i, 4]))
            ax.text(
                coords[:, 0].mean(),
                coords[:, 1].mean(),
                f"{labels_map[int(lbl)]}: {ang_pred:.1f}°/{score:.2f}",
                color="blue",
                fontsize=5,
                ha="center",
                va="center",
                path_effects=[patheffects.withStroke(linewidth=2, foreground="black")],
            )

        save_path = os.path.join(output_dir, os.path.basename(fname))
        fig.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

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
    conf_thres: float = 0.25,
    iou_thres: float = 0.5,
    class_thres: float = 0.5,
    grid_shape: Tuple[int, int] = (3, 3),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Dict[str, Any]:
    """
    Complete inference pipeline:
    1. Prepares anchors
    2. Runs inference and collects predictions
    3. Computes evaluation metrics
    4. Plots visualizations (PR, F1, confusion matrix, boxplots)
    5. Generates qualitative grids and saves per-image results

    Args:
        model (torch.nn.Module): Initialized model architecture.
        test_loader (DataLoader): Dataloader for test set.
        output_dir (str): Directory to store output visualizations.
        device (torch.device): Computation device.
        labels_map (Dict[int, str]): Mapping from class indices to human-readable labels.
        scale_factors (List[float]): Anchor scale factors.
        ratio_factors (List[float]): Anchor ratio factors.
        obb_stats_by_size (Dict): Precomputed OBB stats per resize size.
        conf_thres (float): Confidence threshold for filtering predictions.
        iou_thres (float): IoU threshold to match GT with predictions.
        class_thres (float): Class score threshold for filtering predictions.
        grid_shape (Tuple[int, int]): Rows x Columns in qualitative grid.
        mean (Tuple): Image normalization mean.
        std (Tuple): Image normalization std.

    Returns:
        Dict[str, Any]: Dictionary of figures and computed metrics:
            - mAP
            - PR/F1/Confusion plots
            - IoU and angle boxplots
            - Qualitative grid
    """

    print("[STEP 1] Preparing anchors...")
    resize_size, anchors_xy, _ = prepare_anchors(
        model=model,
        loader=test_loader,
        device=device,
        scale_factors=scale_factors,
        ratio_factors=ratio_factors,
    )

    print("[STEP 2] Running inference...")
    results = run_inference(
        model=model,
        loader=test_loader,
        anchors_xy=anchors_xy,
        resize_size=resize_size,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        class_thres=class_thres,
        device=device,
        labels_map=labels_map,
    )

    print("[STEP 3] Computing metrics and plots...")
    mAP, APs = compute_map_and_pr(results["per_true"], results["per_score"])

    fig_pr = plot_precision_recall(
        per_true=results["per_true"],
        per_score=results["per_score"],
        labels_map=labels_map,
        mAP=mAP,
    )

    fig_cm = plot_confusion_matrix(
        y_true=results["y_true"], y_pred=results["y_pred"], labels_map=labels_map
    )

    iou_data = [
        {"class": labels_map[c], "iou": v}
        for c, vals in results["iou_errs"].items()
        for v in vals
    ]
    fig_iou = plot_boxplots(
        data=iou_data,
        x_field="class",
        y_field="iou",
        title="IoU Distribution per Class",
        labels_map=labels_map,
        y_lim=(0, 1),
    )

    angle_data = [
        {"class": labels_map[c], "error°": v}
        for c, vals in results["angle_errs"].items()
        for v in vals
    ]
    fig_ang = plot_boxplots(
        data=angle_data,
        x_field="class",
        y_field="error°",
        title="Angle-Error Distribution per Class",
        labels_map=labels_map,
        y_lim=(0, 180),
    )

    fig_f1 = plot_f1_vs_threshold(
        all_gts=results["all_gts"],
        all_scores=results["all_scores"],
        all_preds=results["all_preds"],
        labels_map=labels_map,
        default_th=conf_thres,
    )

    fig_grid = plot_qualitative_grid(
        samples=results["samples"],
        labels_map=labels_map,
        grid_shape=grid_shape,
        mean=mean,
        std=std,
    )

    print("[STEP 4] Saving individual prediction images...")
    save_individual_predictions(results["samples"], labels_map, output_dir, mean, std)

    print("[DONE] Inference and reporting completed.")
    return {
        "mAP": mAP,
        "pr_figure": fig_pr,
        "confusion_figure": fig_cm,
        "iou_boxplot_figure": fig_iou,
        "angle_boxplot_figure": fig_ang,
        "f1_threshold_figure": fig_f1,
        "grid_figure": fig_grid,
    }


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
