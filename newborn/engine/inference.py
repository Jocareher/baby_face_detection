import math
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union, Optional, Callable

import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
)

from engine.train import (
    infer_with_rotated_nms,
    get_resize_size,
    generate_anchors_for_training,
    xyxyxyxy2xywhr,
    batch_probiou,
)

from utils.helpers import (
    to_numpy,
    ensure_polygons_42_shape,
    resolve_image_path,
)
from utils.visualize import (
    draw_predictions_on_image,
    write_predictions_txt,
    xywhr_to_poly42_shape,
    scale_polys,
    scale_xywhr_boxes,
    get_oriented_face_crop,
    denormalize_image,
    plot_boxplots,
    plot_child_confusion_matrix,
    plot_confusion_matrix,
    plot_f1_vs_threshold,
    plot_precision_recall,
    plot_qualitative_grid,
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


def run_evaluation(
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
            - viz_payload (Optional[Dict[str, Any]]): Optional payload for visualization, including original image and scaling factors.
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

            # Process each image in the batch
            batch_size = imgs.size(0)
            for b in range(batch_size):
                # Get filename and initialize FP/FN counters
                fname = dataset.file_list[global_idx]
                # Update global index
                global_idx += 1
                # Count of false positives and false negatives in this image
                fp_img, fn_img = 0, 0

                # Load original image for visualization if needed
                viz_payload = None
                # Render original image if specified
                if render_original:
                    # Load original image and compute scaling factors
                    orig_img_np, (sx, sy) = load_original_and_scale(
                        dataset, fname, resize_size
                    )
                    # If loading was successful, prepare payload
                    if orig_img_np is not None:
                        # Visualization payload with original image and scaling
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

            # # Clean up to free memory
            # del imgs, outputs, targets
            # torch.cuda.empty_cache()

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
        cls: (
            average_precision_score(per_true[cls], per_score[cls])
            if sum(per_true[cls]) > 0
            else 0.0
        )
        for cls in per_true
    }
    mAP = float(np.mean(list(APs.values())))
    print(f"[INFO] Computed mAP: {mAP:.4f}")
    return mAP, APs


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
    orig_sizeresolver: Optional[Callable[[str], Optional[Tuple[int, int]]]] = None,
    resize_size: Tuple[int, int] = (640, 640),
) -> None:
    """
    Saves individual visualizations of predictions with ground truth for qualitative analysis.

    This function generates and saves visualization images showing both ground truth and predicted
    oriented bounding boxes (OBBs), with options to use original image resolution and split results
    by error type.

    Args:
        samples: List of tuples containing:
            - Image tensor (normalized)
            - Prediction dictionary with keys:
                - 'polygons': Predicted OBB vertices
                - 'labels': Predicted class labels
                - 'scores': Confidence scores
                - 'boxes': OBB parameters (x,y,w,h,θ)
            - File name
            - Ground truth OBB vertices
            - Ground truth angles
            - Ground truth labels
            - False positive count
            - False negative count
            - Optional visualization payload with:
                - 'orig_img': Original resolution image
                - 'scale': (sx,sy) scaling factors
        labels_map: Mapping from class indices to human-readable labels
        output_dir: Base directory for saving visualizations
        mean: Channel means for image denormalization
        std: Channel standard deviations for denormalization
        split_by_error: Whether to organize outputs into error type subdirectories:
            - tp_only/: Perfect predictions
            - fp/: False positives only
            - fn/: False negatives only
            - fp_fn/: Both error types
        viz_original_res: Whether to render at original image resolution
        orig_sizeresolver: Function to get original (W,H) from filename
        resize_size: Target size used during resizing/inference

    The visualization includes:
        - Ground truth OBBs in dashed green with orange front edge
        - Predicted OBBs in solid blue with red front edge
        - Class labels, angles and confidence scores
        - Optional background in original resolution

    Example directory structure when split_by_error=True:
        output_dir/
        ├── tp_only/
        │   ├── image1.jpg
        │   └── image2.jpg
        ├── fp/
        │   └── image3.jpg
        ├── fn/
        │   └── image4.jpg
        └── fp_fn/
            └── image5.jpg
    """
    # Convert output directory to Path object for easier manipulation
    base_dir = Path(output_dir)

    # Create subdirectories for different error types if splitting is enabled
    if split_by_error:
        for sub in ("tp_only", "fp", "fn", "fp_fn"):
            (base_dir / sub).mkdir(parents=True, exist_ok=True)
    else:
        base_dir.mkdir(parents=True, exist_ok=True)

    # Process each sample
    for sample in samples:
        # Handle samples with or without visualization payload
        if len(sample) == 9:
            img_t, out, fname, gt_b, gt_a, gt_l, fp_img, fn_img, viz = sample
        else:
            img_t, out, fname, gt_b, gt_a, gt_l, fp_img, fn_img = sample
            viz = None

        # Determine background image and scaling factors
        if viz is not None and viz.get("orig_img", None) is not None:
            # Use provided original resolution image if available
            base_img = viz["orig_img"]  # np.uint8 (H0, W0, 3)
            sx, sy = viz["scale"]
        else:
            # Fallback to denormalized tensor and try upscaling if requested
            base_img = denormalize_image(img_t, mean=mean, std=std)  # (Hr, Wr, 3)
            sx, sy = 1.0, 1.0
            if viz_original_res and orig_sizeresolver is not None:
                wh = orig_sizeresolver(fname)
                if wh is not None:
                    W0, H0 = wh
                    Wr, Hr = resize_size
                    sx = float(W0) / float(Wr)
                    sy = float(H0) / float(Hr)
                    try:
                        from PIL import Image

                        base_img = np.asarray(
                            Image.fromarray(base_img).resize((W0, H0))
                        )
                    except Exception:
                        sx, sy = 1.0, 1.0  # fallback to 640x640 if resize fails

        # Setup matplotlib figure to match image dimensions exactly
        H_out, W_out = int(base_img.shape[0]), int(base_img.shape[1])
        dpi = 100
        fig = plt.figure(figsize=(W_out / dpi, H_out / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  # use full canvas without margins

        # Display base image with correct alignment and no interpolation artifacts
        ax.imshow(base_img, extent=(0, W_out, H_out, 0), interpolation="nearest")
        ax.set_xlim(0, W_out)
        ax.set_ylim(H_out, 0)  # invert Y axis for image coordinates
        ax.axis("off")

        # Draw ground truth OBBs (dashed green with orange front edge)
        for pts, ang, lbl in zip(gt_b, gt_a, gt_l):
            coords = pts.detach().cpu().view(4, 2).numpy()
            # Scale coordinates to original resolution if needed
            coords[:, 0] *= sx
            coords[:, 1] *= sy
            # Draw OBB polygon
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
            # Draw front edge in orange
            ax.plot(coords[[0, 1], 0], coords[[0, 1], 1], color="orange", linewidth=2)

            # Add class label and angle at bottom-right with green background
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

        # Draw predicted OBBs (solid blue with red front edge)
        for i, (pts, lbl, score) in enumerate(
            zip(out["polygons"], out["labels"], out["scores"])
        ):
            coords = pts.cpu().view(4, 2).numpy()
            # Scale coordinates to original resolution if needed
            coords[:, 0] *= sx
            coords[:, 1] *= sy
            # Draw OBB polygon
            ax.add_patch(
                patches.Polygon(
                    coords,
                    closed=True,
                    fill=False,
                    edgecolor="#004080",  # Dark blue for predictions
                    linewidth=1.5,
                )
            )
            # Draw front edge in dark red
            ax.plot(
                coords[[0, 1], 0], coords[[0, 1], 1], color="#800000", linewidth=1.5
            )

            # Add class label, angle and confidence at top-left with blue background
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

        # Determine output subdirectory based on error types
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
                subdir = "tp_only"  # Perfect predictions (true positives only)

            save_dir = base_dir / subdir
            save_dir.mkdir(exist_ok=True, parents=True)

        # Save visualization without padding and close to free memory
        fig.savefig(
            save_dir / Path(fname).name, dpi=dpi, bbox_inches=None, pad_inches=0
        )
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
    results = run_evaluation(
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
    resolver = build_image_sizeresolver(test_loader.dataset)
    save_individual_predictions(
        samples=results["samples"],
        labels_map=labels_map,
        output_dir=predictions_dir,
        mean=mean,
        std=std,
        split_by_error=True,
        viz_original_res=render_original,
        orig_sizeresolver=resolver,
        resize_size=resize_size,
    )
    print("[DONE] Inference and reporting completed.")

    return {"mAP": mAP, "APs": APs}


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


def build_image_sizeresolver(dataset, images_subdir: str = "images") -> callable:
    """
    Builds a function that resolves original image dimensions from a dataset.

    Creates a callable that attempts to load and get dimensions of an original image,
    first trying the filename directly and then searching in the dataset's image directory
    with different extensions.

    Args:
        dataset: Dataset object with root_dir and split attributes defining image location
        images_subdir (str, optional): Subdirectory name containing images. Defaults to "images"

    Returns:
        callable: Function that takes a filename (stem or path) and returns:
            - Tuple[int, int]: Original (width, height) if image is found
            - None: If image cannot be found or opened

    Example:
        >>> resolver = build_image_sizeresolver(dataset)
        >>> size = resolver("image001.jpg")  # Returns (1024, 768) or None
    """
    # Construct path to images directory from dataset attributes
    root = Path(dataset.root_dir) / dataset.split / images_subdir

    # Common image file extensions to try if bare filename is provided
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def resolve(fname: str) -> tuple[int, int]:
        p = Path(fname)

        # Strategy 1: Try direct path if it's a complete filepath
        if p.is_file():
            with Image.open(p) as im:
                return im.size  # Returns (width, height)

        # Strategy 2: Try different extensions in dataset image directory
        # Extract stem (filename without extension) to try with different extensions
        stem = p.stem if p.suffix else p.name
        for ext in exts:
            candidate = root / f"{stem}{ext}"
            if candidate.exists():
                with Image.open(candidate) as im:
                    return im.size

        # Return None if image cannot be found or opened
        return None

    return resolve


def load_original_and_scale(dataset, fname: str, resize_size: Tuple[int, int]):
    """
    Load the original image and calculate the scaling factors (sx, sy) relative to the resized dimensions.

    Args:
        dataset: The dataset object containing image metadata and paths.
        fname (str): The filename of the image to load.
        resize_size (Tuple[int, int]): The target size (W_r, H_r) to which the image was resized.

    Returns:
        Tuple[np.ndarray, Tuple[float, float]]: A tuple containing:
            - np_img_rgb (np.ndarray): The original image in RGB format as a NumPy array.
            - (sx, sy) (Tuple[float, float]): Scaling factors for width and height.
              Returns (1.0, 1.0) if the image cannot be loaded.
    """
    # Build a function to resolve the original image dimensions from the dataset
    resolver = build_image_sizeresolver(dataset)

    # Get the original width and height of the image
    wh = resolver(fname)
    if wh is None:
        return None, (
            1.0,
            1.0,
        )  # Return None and default scaling factors if image not found

    W0, H0 = wh  # Original image dimensions (width, height)
    Wr, Hr = resize_size  # Resized image dimensions
    sx = float(W0) / float(Wr)  # Calculate scaling factor for width
    sy = float(H0) / float(Hr)  # Calculate scaling factor for height

    # Load the original image from the dataset's images directory
    root = Path(dataset.root_dir) / dataset.split / "images"
    p = Path(fname)

    # Check if the provided path is a valid file
    if not p.is_file():
        # If not, try to find the image with common extensions
        stem = p.stem if p.suffix else p.name
        for e in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            cand = root / f"{stem}{e}"
            if cand.exists():
                p = cand  # Update path to the found image
                break

    try:
        # Open the image and convert it to RGB format
        with Image.open(p) as im:
            im = im.convert("RGB")
            np_img = np.asarray(im)  # Convert the image to a NumPy array
            return np_img, (sx, sy)  # Return the image and scaling factors
    except Exception:
        return None, (
            1.0,
            1.0,
        )  # Return None and default scaling factors if an error occurs


@torch.inference_mode()
def export_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    anchors_xy: torch.Tensor,
    resize_size: Tuple[int, int],  # (W, H) used by model
    face_thres: float,
    iou_thres: float,
    class_thres: float,
    baby_thres: float,
    device: torch.device,
    labels_map: Dict[int, str],
    out_dir: Path,
    output_scale: str = "original",  # "original" | "resized"
) -> None:
    """
    Export model predictions as annotated images and text files.

    This function performs inference on a dataset and saves:
    1. Annotated images with detected faces and their orientations
    2. Text files with detection coordinates and metadata
    3. Cropped face images per detection

    The output can be saved in two coordinate scales:
    - "original": Native resolution of input images
    - "resized": Model input resolution (e.g., 640x640)

    Args:
        model: The RetinaBabyFace detection model
        loader: DataLoader providing images (and optionally labels)
        anchors_xy: Anchor box coordinates tensor
        resize_size: Model input size as (width, height)
        face_thres: Face detection confidence threshold
        iou_thres: IoU threshold for NMS
        class_thres: Classification confidence threshold
        baby_thres: Baby face confidence threshold
        device: Torch device to run inference on
        labels_map: Dictionary mapping class indices to names
        out_dir: Base output directory path
        output_scale: Whether to save in "original" or "resized" coordinates

    Outputs:
        - out_dir/images/: Directory with annotated images
        - out_dir/labels/: Directory with text files containing detections
        - out_dir/crops/: Directory with cropped face images per class
    """

    # Create output directories
    out_imgs = Path(out_dir) / "images"
    out_lbls = Path(out_dir) / "labels"
    out_crops = Path(out_dir) / "crops"
    for d in (out_imgs, out_lbls, out_crops):
        d.mkdir(parents=True, exist_ok=True)

    # Setup model and anchors
    model.eval()
    anchors_xy = anchors_xy.to(device, non_blocking=True)
    Wr, Hr = resize_size
    nms_image_size = (Hr, Wr)  # NMS expects (H,W) format

    dataset = loader.dataset

    # Statistics counters
    processed = 0  # Total images processed
    saved = 0  # Successfully saved pairs
    empty_batches = 0  # Batches with no valid images
    no_dets = 0  # Images with no detections
    errors = 0  # Failed operations

    # Print configuration
    tqdm.write(f"🧠  Inference on device: {device}")
    tqdm.write(
        f"Dataloader: {len(loader)} batches | batch_size={getattr(loader, 'batch_size', '?')}"
    )
    tqdm.write(f"Output dir: {out_dir}  →  images/, labels/")
    tqdm.write(f"Resize size (W,H): {resize_size} | NMS uses (H,W)={nms_image_size}")
    tqdm.write(f"Output scale: {output_scale}")

    with tqdm(total=len(loader), desc="⚙️  Batches", unit="batch") as pbar_batches:
        global_idx = 0

        for batch in loader:
            # Move images to device
            imgs = batch["image"].to(device, non_blocking=True)
            if imgs.numel() == 0:
                empty_batches += 1
                pbar_batches.update(1)
                continue

            # Run inference and NMS
            try:
                outputs = infer_with_rotated_nms(
                    model_or_preds=model,
                    images=imgs,
                    anchors_xy=anchors_xy,
                    image_size=nms_image_size,
                    face_thres=face_thres,
                    baby_thres=baby_thres,
                    iou_thres=iou_thres,
                    class_thres=class_thres,
                )
            except Exception as e:
                errors += 1
                tqdm.write(f"Inference error in batch: {e}")
                pbar_batches.update(1)
                continue

            # Process each image in batch
            B = imgs.size(0)
            with tqdm(
                total=B, desc="   Images", leave=False, unit="img"
            ) as pbar_imgs:
                for b in range(B):
                    processed += 1

                    # Get robust path for image
                    p = resolve_image_path(batch, b, global_idx, dataset=dataset)
                    stem, ext = p.stem, (p.suffix if p.suffix else ".jpg")

                    # Load base image and compute scale factors
                    try:
                        if output_scale == "original" and p.exists():
                            # Load original image
                            with Image.open(p) as im:
                                im = im.convert("RGB")
                                W0, H0 = im.size
                                base_img = np.asarray(im)
                            sx, sy = float(W0) / float(Wr), float(H0) / float(Hr)
                        else:
                            # Use resized tensor or fallback
                            base_img = denormalize_image(imgs[b])
                            sx, sy = 1.0, 1.0
                            if not ext:
                                ext = ".jpg"
                    except Exception as e:
                        errors += 1
                        tqdm.write(f"Could not prepare base image for {p}: {e}")
                        pbar_imgs.update(1)
                        global_idx += 1
                        continue

                    # Extract predictions
                    try:
                        out_b = outputs[b]
                        boxes_np = to_numpy(
                            out_b.get("boxes")
                        )  # (N,5) -> cx,cy,w,h,theta
                        labels_np = to_numpy(out_b.get("labels"))
                        scores_np = to_numpy(out_b.get("scores"))
                        polys_np = to_numpy(out_b.get("polygons"))  # (N,8) or (N,4,2)

                        # Normalize/reconstruct polygons if needed
                        polys_42 = ensure_polygons_42_shape(polys_np)
                        if (
                            (polys_42 is None or polys_42.size == 0)
                            and boxes_np is not None
                            and boxes_np.size > 0
                        ):
                            N = boxes_np.shape[0]
                            polys_42 = np.stack(
                                [xywhr_to_poly42_shape(*boxes_np[i]) for i in range(N)],
                                axis=0,
                                dtype=np.float32,
                            )
                    except Exception as e:
                        errors += 1
                        tqdm.write(f"Postprocess error for {p}: {e}")
                        pbar_imgs.update(1)
                        global_idx += 1
                        continue

                    # Scale coordinates to target output size
                    if output_scale == "original" and (sx != 1.0 or sy != 1.0):
                        polys_for_img = scale_polys(polys_42, sx, sy)
                        boxes_for_txt = (
                            scale_xywhr_boxes(boxes_np, sx, sy)
                            if boxes_np is not None
                            else None
                        )
                    else:
                        polys_for_img = polys_42
                        boxes_for_txt = boxes_np

                    # Save results
                    try:
                        if polys_for_img is not None and polys_for_img.size > 0:
                            angles = (
                                boxes_np[:, 4]
                                if (boxes_np is not None and boxes_np.size > 0)
                                else np.zeros((0,), dtype=np.float32)
                            )
                            lbls = (
                                labels_np
                                if labels_np is not None
                                else np.zeros((polys_for_img.shape[0],), dtype=np.int64)
                            )
                            scrs = (
                                scores_np
                                if scores_np is not None
                                else np.zeros(
                                    (polys_for_img.shape[0],), dtype=np.float32
                                )
                            )
                            painted = draw_predictions_on_image(
                                base_img=base_img,
                                polygons_xy=polys_for_img,
                                labels=lbls,
                                scores=scrs,
                                angles_rad=angles,
                                labels_map=labels_map,
                            )
                        else:
                            painted = base_img
                            no_dets += 1

                        Image.fromarray(painted).save(out_imgs / f"{stem}{ext}")

                        write_predictions_txt(
                            out_labels_dir=out_lbls,
                            stem=stem,
                            boxes_xywhr=boxes_for_txt,
                            polygons_42=polys_for_img,
                            labels=labels_np,
                            scores=scores_np,
                        )
                        saved += 1

                    except Exception as e:
                        errors += 1
                        tqdm.write(f"Saving error for {p}: {e}")

                    if polys_for_img is not None and polys_for_img.size > 0:
                        # Model input size (width, height) used for the face crops (e.g., 640x640)
                        Wr_target, Hr_target = resize_size
                        for j in range(polys_for_img.shape[0]):
                            # polygon for this detection as (4,2) float32 array
                            poly = polys_for_img[j].astype(np.float32)

                            # angle from the detected box in radians; fallback to 0.0 if not present
                            theta = (
                                float(boxes_for_txt[j, 4])
                                if (
                                    boxes_for_txt is not None and boxes_for_txt.size > 0
                                )
                                else 0.0
                            )

                            # Extract an oriented crop of the face using the polygon and rotation angle.
                            # - base_img: source image (numpy array HxWxC)
                            # - poly42: polygon in 4x2 format (corner coordinates)
                            # - angle_rad: rotation to apply (radians)
                            # - pivot: reference point for rotation ("tl" = top-left, or "center")
                            # - crop_out_wh: output crop size (W,H)
                            # - border_mode: how to fill borders when rotating ("replicate"/"black"/"white")
                            # - scale_crop: scale factor to add context (>1.0 adds margin)
                            crop640 = get_oriented_face_crop(
                                base_img=base_img,
                                poly42=poly,
                                angle_rad=theta,
                                pivot="tl",  # or "center" if preferred
                                crop_out_wh=(Wr_target, Hr_target),
                                border_mode="replicate",  # "replicate" (no black borders) | "black" | "white"
                                scale_crop=1.0,  # 1.0 = no margin; >1.0 adds context
                            )
                            # Skip if crop extraction failed
                            if crop640 is None:
                                continue

                            # Determine class index and name for this detection (fallback to 0)
                            cls_idx = (
                                int(labels_np[j])
                                if (labels_np is not None and labels_np.size > j)
                                else 0
                            )
                            cls_name = labels_map.get(cls_idx, str(cls_idx))
                            # Prepare class-specific output directory for crops using class name
                            cls_dir = Path(out_dir) / "crops" / cls_name
                            cls_dir.mkdir(parents=True, exist_ok=True)
                            # Save the crop as a JPEG with an index in the filename
                            Image.fromarray(crop640).save(
                                cls_dir / f"{stem}_{j:02d}.jpg"
                            )

                    pbar_imgs.update(1)
                    global_idx += 1

            pbar_batches.update(1)

    # Print summary statistics
    tqdm.write("Export complete")
    tqdm.write(f"   • Processed images : {processed}")
    tqdm.write(f"   • Saved (img+txt)  : {saved}")
    tqdm.write(f"   • No detections    : {no_dets}")
    tqdm.write(f"   • Empty batches    : {empty_batches}")
    tqdm.write(f"   • Errors           : {errors}")
    tqdm.write(f"Images: {out_imgs}")
    tqdm.write(f"Labels: {out_lbls}")
    tqdm.write(f"Crops : {out_crops}")
