import os
import math
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
from matplotlib.patches import Polygon as MplPolygon
from torch.utils.data import DataLoader
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
    get_base_obb_stats,
    generate_anchors_for_training,
    xyxyxyxy2xywhr,
    batch_probiou,
    denormalize_image,
)

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
    obb_stats: Dict[Tuple[int, int], Dict[str, float]],
) -> Tuple[Tuple[int, int], torch.Tensor, torch.Tensor]:
    """
    Prepares anchors for inference using the base OBB statistics and given scale/ratio factors.

    Args:
        model (torch.nn.Module): The model used for feature extraction to generate anchors.
        loader (DataLoader): DataLoader to estimate image resize size.
        device (torch.device): Device for tensor creation (usually 'cuda' or 'cpu').
        scale_factors (List[float]): List of scaling factors for anchor generation.
        ratio_factors (List[float]): List of aspect ratio factors for anchors.
        obb_stats (Dict[Tuple[int, int], Dict[str, float]]): Dictionary with average OBB stats per image size.

    Returns:
        Tuple containing:
            - resize_size (Tuple[int, int]): Target size used to resize images.
            - anchors_xy (torch.Tensor): Anchors in (x, y) format for initial location.
            - anchors_xywhr (torch.Tensor): Anchors in (x, y, w, h, θ) format.
    """
    # Get target resize size from the dataset
    resize_size = get_resize_size(loader)

    # Extract base average size and aspect ratio from precomputed stats
    base_size, base_ratio = get_base_obb_stats(resize_size, obb_stats)

    # Generate anchors in both formats
    anchors_xy, anchors_xywhr = generate_anchors_for_training(
        model=model,
        resize_size=resize_size,
        device=device,
        base_size=base_size,
        base_ratio=base_ratio,
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
    device: torch.device,
    labels_map: Dict[int, str],
) -> Dict[str, Any]:
    """
    Performs inference on the given dataset using the model and rotated NMS.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        loader (DataLoader): DataLoader providing batches of test data.
        anchors_xy (torch.Tensor): Anchors in (x, y) format used for predictions.
        resize_size (Tuple[int, int]): Size to which images are resized.
        conf_thres (float): Confidence threshold for filtering detections.
        iou_thres (float): IoU threshold for determining TP/FP.
        device (torch.device): Computation device (e.g. 'cuda' or 'cpu').
        labels_map (Dict[int, str]): Mapping from class indices to readable labels.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - detection and angle errors
            - per-class true/score pairs for PR/mAP
            - confusion matrix data
            - qualitative samples for visualization
    """

    # Initialize structures for metrics
    per_true, per_score = {c: [] for c in labels_map}, {c: [] for c in labels_map}
    iou_errs, angle_errs = {c: [] for c in labels_map}, {c: [] for c in labels_map}
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in labels_map}

    y_true, y_pred = [], []
    all_scores, all_preds, all_gts = [], [], []
    samples = []

    dataset = loader.dataset
    global_idx = 0

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference"):
            imgs = batch["image"].to(device)
            targets = batch["target"]

            # Get predictions using rotated NMS
            outputs = infer_with_rotated_nms(
                model, imgs, anchors_xy, resize_size, conf_thres, iou_thres
            )

            batch_size = imgs.size(0)
            for b in range(batch_size):
                fname = dataset.file_list[global_idx]
                global_idx += 1

                # Extract valid GT elements
                valid_mask = targets["valid_mask"][b]
                gt_boxes = targets["boxes"][b][valid_mask]
                gt_angles = targets["angles"][b][valid_mask].view(-1)
                gt_labels = targets["class_idx"][b][valid_mask]

                # Convert to xywhr format
                gt_xywhr = xyxyxyxy2xywhr(
                    gt_boxes, gt_angles.unsqueeze(-1), resize_size
                ).to(device)

                # Store CPU copy for qualitative visualization
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

                # Extract predictions
                pred_boxes = outputs[b]["boxes"].to(device)
                pred_scores = outputs[b]["scores"].to(device)
                pred_labels = outputs[b]["labels"].to(device)

                num_gt, num_pred = gt_xywhr.size(0), pred_boxes.size(0)

                # Compute IoU between GT and predictions
                iou_matrix = (
                    batch_probiou(gt_xywhr, pred_boxes)
                    if num_gt > 0 and num_pred > 0
                    else torch.zeros(num_gt, num_pred, device=device)
                )
                matched = torch.zeros(num_pred, dtype=torch.bool, device=device)

                # Match each GT box to best prediction
                for i in range(num_gt):
                    cls = int(gt_labels[i].item())

                    if num_pred == 0:
                        # No predictions available
                        stats[cls]["fn"] += 1
                        for c in labels_map:
                            per_true[c].append(int(c == cls))
                            per_score[c].append(0.0)
                        y_true.append(cls)
                        y_pred.append(-1)
                        all_gts.append(cls)
                        all_scores.append(0.0)
                        all_preds.append(-1)
                        continue

                    best_iou, best_j = iou_matrix[i].max(0)
                    is_match = best_iou >= iou_thres

                    if is_match:
                        stats[cls]["tp"] += 1
                        iou_errs[cls].append(best_iou.item())
                        angle_diff = abs(
                            (pred_boxes[best_j, 4] - gt_angles[i]) * 180 / math.pi
                        )
                        angle_errs[cls].append(angle_diff.item())
                        matched[best_j] = True
                    else:
                        stats[cls]["fn"] += 1

                    for c in labels_map:
                        per_true[c].append(int(c == cls))
                        per_score[c].append(
                            pred_scores[best_j].item() if is_match else 0.0
                        )

                    y_true.append(cls)
                    y_pred.append(int(pred_labels[best_j].item()) if is_match else -1)
                    all_gts.append(cls)
                    all_scores.append(pred_scores[best_j].item() if is_match else 0.0)
                    all_preds.append(
                        int(pred_labels[best_j].item()) if is_match else -1
                    )

                # Count unmatched predictions as false positives
                for k in range(num_pred):
                    if not matched[k]:
                        cls = int(pred_labels[k].item())
                        stats[cls]["fp"] += 1

            # Free memory after each batch
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
        "all_scores": all_scores,
        "all_preds": all_preds,
        "all_gts": all_gts,
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

        prec_s = smooth_curve(prec, sigma)
        rec_s = smooth_curve(rec, sigma)
        ap = average_precision_score(y_t, y_s) if y_t.sum() > 0 else 0.0

        ax.plot(rec_s, prec_s, lw=2, label=f"{labels_map[cls]} {ap:.3f}")

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
    Plots F1 Score vs. confidence threshold for each class.

    Args:
        all_gts (List[int]): Ground truth class labels.
        all_scores (List[float]): Prediction confidence scores.
        all_preds (List[int]): Predicted class labels.
        labels_map (Dict[int, str]): Class index to name mapping.
        default_th (float): Default threshold to highlight.
        n_steps (int): Number of threshold steps between [0, 1].
        sigma (float): Smoothing factor for the F1 curve.

    Returns:
        matplotlib.figure.Figure: F1 vs. threshold plot.
    """
    thresholds = np.linspace(0.0, 1.0, n_steps)
    y_true = np.array(all_gts)
    classes = list(labels_map.keys())

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("F1 vs. Confidence Threshold", fontsize=13)

    for cls in classes:
        f1s = []
        for t in thresholds:
            # Convert confidence to prediction using threshold
            y_pred = [lbl if sc >= t else -1 for sc, lbl in zip(all_scores, all_preds)]
            # Compute F1 score for the current class
            f1_val = f1_score(
                y_true, y_pred, labels=classes, average=None, zero_division=0
            )
            f1s.append(f1_val[classes.index(cls)])

        f1s = np.array(f1s)
        f1_s = smooth_curve(f1s, sigma)
        ax.plot(thresholds, f1_s, lw=2, label=f"{labels_map[cls]} {f1_s.mean():.3f}")

        # Mark the best point
        best_i = f1_s.argmax()
        ax.axvline(thresholds[best_i], linestyle="--", lw=1)
        ax.scatter([thresholds[best_i]], [f1_s[best_i]], s=50, zorder=3)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence Threshold", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1.0), fontsize=12, frameon=False)
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
                    pts_np, closed=True, fill=False, edgecolor="blue", linewidth=2
                )
            )
            ax.plot(pts_np[[0, 1], 0], pts_np[[0, 1], 1], color="red", linewidth=2)
            cen = pts_np.mean(axis=0)
            ax.text(
                cen[0],
                cen[1],
                f"{labels_map[int(cls)]}: {math.degrees(angle):.1f}°",
                color="blue",
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
                    edgecolor="green",
                    linewidth=1.5,
                    linestyle="--",
                )
            )
            ax.plot(pts_np[[0, 1], 0], pts_np[[0, 1], 1], color="orange", linewidth=1.5)
            cen = pts_np.mean(axis=0)
            ang = math.degrees(float(out["boxes"][i, 4]))
            ax.text(
                cen[0],
                cen[1],
                f"{labels_map[int(lbl)]}: {ang:.1f}°/{score:.2f}",
                color="green",
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
                    coords, closed=True, fill=False, edgecolor="blue", linewidth=2
                )
            )
            ax.plot(coords[[0, 1], 0], coords[[0, 1], 1], color="red", linewidth=2)
            ax.text(
                coords[:, 0].mean(),
                coords[:, 1].mean(),
                f"{labels_map[int(lbl)]}: {math.degrees(float(ang)):.1f}°",
                color="blue",
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
                    edgecolor="green",
                    linewidth=1.5,
                    linestyle="--",
                )
            )
            ax.plot(coords[[0, 1], 0], coords[[0, 1], 1], color="orange", linewidth=1.5)
            ang_pred = math.degrees(float(out["boxes"][i, 4]))
            ax.text(
                coords[:, 0].mean(),
                coords[:, 1].mean(),
                f"{labels_map[int(lbl)]}: {ang_pred:.1f}°/{score:.2f}",
                color="green",
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
    checkpoint_path: str,
    test_loader: DataLoader,
    output_dir: Union[str, Path],
    device: torch.device,
    labels_map: Dict[int, str],
    scale_factors: List[float],
    ratio_factors: List[float],
    obb_stats_by_size: Dict[Tuple[int, int], Dict[str, float]],
    conf_thres: float = 0.25,
    iou_thres: float = 0.5,
    grid_shape: Tuple[int, int] = (3, 3),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Dict[str, Any]:
    """
    Complete inference pipeline:
    1. Loads model checkpoint
    2. Prepares anchors
    3. Runs inference and collects predictions
    4. Computes evaluation metrics
    5. Plots visualizations (PR, F1, confusion matrix, boxplots)
    6. Generates qualitative grids and saves per-image results

    Args:
        model (torch.nn.Module): Initialized model architecture.
        checkpoint_path (str): Path to trained checkpoint file (.pt or .pth).
        test_loader (DataLoader): Dataloader for test set.
        output_dir (str): Directory to store output visualizations.
        device (torch.device): Computation device.
        labels_map (Dict[int, str]): Mapping from class indices to human-readable labels.
        scale_factors (List[float]): Anchor scale factors.
        ratio_factors (List[float]): Anchor ratio factors.
        obb_stats_by_size (Dict): Precomputed OBB stats per resize size.
        conf_thres (float): Confidence threshold for filtering predictions.
        iou_thres (float): IoU threshold to match GT with predictions.
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    print("[STEP 1] Loading checkpoint...")
    load_model_checkpoint(model, checkpoint_path, device)

    print("[STEP 2] Preparing anchors...")
    resize_size, anchors_xy, _ = prepare_anchors(
        model=model,
        loader=test_loader,
        device=device,
        scale_factors=scale_factors,
        ratio_factors=ratio_factors,
        obb_stats=obb_stats_by_size,
    )

    print("[STEP 3] Running inference...")
    results = run_inference(
        model=model,
        loader=test_loader,
        anchors_xy=anchors_xy,
        resize_size=resize_size,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        device=device,
        labels_map=labels_map,
    )

    print("[STEP 4] Computing metrics and plots...")
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

    print("[STEP 5] Saving individual prediction images...")
    save_individual_predictions(results["samples"], labels_map, pred_dir, mean, std)

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
