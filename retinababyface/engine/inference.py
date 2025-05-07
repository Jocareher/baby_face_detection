import os
import math
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from loss.utils import batch_probiou, xyxyxyxy2xywhr
from utils.visualize import denormalize_image
from engine.train import (
    infer_with_rotated_nms,
    generate_anchors_for_training,
    get_resize_size,
    get_base_obb_stats,
)


def inference(
    model: torch.nn.Module,
    checkpoint_path: str,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    labels_map: Dict[int, str],
    scale_factors: List[float],
    ratio_factors: List[float],
    obb_stats_by_size: Dict[Tuple[int, int], Dict[str, float]],
    conf_thres: float = 0.25,
    iou_thres: float = 0.5,
    grid_shape: Tuple[int, int] = (3, 3),
) -> Dict[str, plt.Figure]:
    """
    Runs full evaluation of a pretrained RetinaBabyFace model over a test set.

    This function loads the model from a checkpoint, generates anchors for the test image size,
    performs inference using rotated NMS, and computes visual and quantitative evaluation:

    - Precision–Recall (PR) curves for each class
    - Confusion matrix including background (-1)
    - A qualitative grid of sample predictions with angle and class labels

    Ground truth is extracted from the dataset, and each prediction is compared using
    probabilistic IoU (batch_probiou). A prediction is considered a match if IoU ≥ iou_thres.

    Args:
        model (nn.Module): The RetinaBabyFace model instance.
        checkpoint_path (str): Path to the pretrained checkpoint (.pt file).
        test_loader (DataLoader): Dataloader containing the test split.
        device (torch.device): Device to run inference on (e.g., 'cuda' or 'cpu').
        labels_map (Dict[int, str]): Mapping from class indices to human-readable names.
        scale_factors (List[float]): Anchor scale factors.
        ratio_factors (List[float]): Anchor ratio factors.
        obb_stats_by_size (Dict[Tuple[int,int], Dict[str,float]]): Precomputed anchor stats by image size.
        conf_thres (float): Confidence threshold for predictions.
        iou_thres (float): IoU threshold for a positive match during evaluation.
        grid_shape (Tuple[int,int]): Rows and columns for qualitative visualization grid.

    Returns:
        Dict[str, plt.Figure]: A dictionary with:
            - 'pr_figure': Precision–Recall curves figure.
            - 'confusion_figure': Confusion matrix figure.
            - 'grid_figure': Visualization of predictions vs ground truth.
    """

    # 1) Load checkpoint and model weights
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()

    # 2) Generate anchors based on test image size
    resize_size = get_resize_size(test_loader)
    base_size, base_ratio = get_base_obb_stats(resize_size, obb_stats_by_size)
    anchors_xy, anchors_xywhr = generate_anchors_for_training(
        model, resize_size, device, base_size, base_ratio, scale_factors, ratio_factors
    )

    # 3) Initialize containers for metrics and qualitative results
    per_class_true = {c: [] for c in labels_map}
    per_class_score = {c: [] for c in labels_map}
    y_pred_cls, y_true_cls = [], []
    samples = []  # Will store examples for the qualitative grid

    dataset = test_loader.dataset
    sample_idx = 0

    # 4) Inference loop
    with torch.inference_mode():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            targets = batch["target"]
            outs = infer_with_rotated_nms(
                model,
                imgs,
                anchors_xy,
                resize_size,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
            )
            B = imgs.size(0)

            for b in range(B):
                fname = dataset.file_list[sample_idx]
                sample_idx += 1

                valid = targets["valid_mask"][b]
                gt_lbls = targets["class_idx"][b][valid]
                gt_polys = targets["boxes"][b][valid]
                gt_angs = targets["angles"][b][valid].squeeze(-1)

                # Convert GT to (cx, cy, w, h, angle) for metrics
                gt_xywhr = (
                    xyxyxyxy2xywhr(gt_polys, gt_angs.unsqueeze(-1), resize_size)
                    .cpu()
                    .numpy()
                )
                gt_lbls_np = gt_lbls.cpu().numpy()

                if gt_polys.ndim == 1:
                    gt_polys = gt_polys.unsqueeze(0)
                gt_angs = gt_angs.view(-1)

                # Store sample for final qualitative visualization
                samples.append(
                    (
                        imgs[b].cpu(),
                        outs[b],
                        fname,
                        gt_polys.cpu(),
                        gt_angs.cpu(),
                        gt_lbls.cpu(),
                    )
                )

                # === A) Precision–Recall Curve Calculation ===
                pred = outs[b]
                p_boxes = pred["boxes"].cpu().numpy()
                p_scores = pred["scores"].cpu().numpy()
                p_lbls = pred["labels"].cpu().numpy().astype(int)

                for c in labels_map:
                    for g_lbl, g_box in zip(gt_lbls_np, gt_xywhr):
                        per_class_true[c].append(int(g_lbl == c))
                        if p_boxes.size:
                            ious = (
                                batch_probiou(
                                    torch.from_numpy(p_boxes).to(device),
                                    torch.from_numpy(g_box[None]).to(device),
                                )
                                .cpu()
                                .numpy()
                                .ravel()
                            )
                            if (ious >= iou_thres).any():
                                per_class_score[c].append(
                                    float(p_scores[ious >= iou_thres].max())
                                )
                            else:
                                per_class_score[c].append(0.0)
                        else:
                            per_class_score[c].append(0.0)

                # === B) Confusion Matrix Calculation ===
                for g_lbl, g_box in zip(gt_lbls_np, gt_xywhr):
                    if p_boxes.size:
                        ious = (
                            batch_probiou(
                                torch.from_numpy(p_boxes).to(device),
                                torch.from_numpy(g_box[None]).to(device),
                            )
                            .cpu()
                            .numpy()
                            .ravel()
                        )
                        j = int(np.argmax(ious))
                        y_pred_cls.append(
                            int(p_lbls[j]) if ious[j] >= iou_thres else -1
                        )
                    else:
                        y_pred_cls.append(-1)
                    y_true_cls.append(int(g_lbl))

    # 5) Plot Precision–Recall curves
    plt.close("all")
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))

    for c, name in labels_map.items():
        y_t = np.array(per_class_true[c])
        y_s = np.array(per_class_score[c])

        if y_t.sum() == 0:
            ax_pr.step([0.0, 1.0], [1.0, 1.0], where="post", label=f"{name} (npos=0)")
            continue

        precision, recall, _ = precision_recall_curve(y_t, y_s)
        auc = np.trapz(precision, recall)
        ax_pr.step(recall, precision, where="post", label=f"{name} {auc:.3f}")

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–Recall Curves")
    ax_pr.legend(loc="upper right", fontsize=8)
    fig_pr.tight_layout()

    # 6) Plot Confusion Matrix (including background class -1)
    cm_labels = list(labels_map.keys()) + [-1]
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=cm_labels)
    display_lbls = [labels_map.get(c, "BG") for c in cm_labels]
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_lbls)
    disp.plot(
        ax=ax_cm, cmap="Blues", colorbar=False, xticks_rotation=45, values_format="d"
    )
    ax_cm.set_title("Confusion Matrix", pad=20)
    fig_cm.tight_layout()

    # 7) Visual Prediction Grid
    rows, cols = grid_shape
    fig_grid, axes = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 4), facecolor="white"
    )
    axes = axes.flatten()

    for ax, (img_t, pred, fname, gt_polys, gt_angs, gt_lbls) in zip(
        axes, samples[: rows * cols]
    ):
        ax.imshow(denormalize_image(img_t))
        ax.set_title(os.path.basename(fname), fontsize=7, color="#333")
        ax.axis("off")
        ax.set_aspect("equal")

        # GT polygons
        for poly, ang, lbl in zip(gt_polys, gt_angs, gt_lbls):
            pts = poly.view(4, 2).numpy()
            ax.add_patch(
                MplPolygon(pts, closed=True, fill=False, edgecolor="#0055FF", lw=2)
            )
            ax.plot(
                [pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]], color="#FF3333", lw=2
            )
            ax.text(
                pts[:, 0].mean(),
                pts[:, 1].mean(),
                f"{labels_map[int(lbl)]}\n{math.degrees(float(ang)):.1f}°",
                color="#0055FF",
                fontsize=6,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#0055FF", lw=0.5),
            )

        # Predicted polygons
        p_polys = pred["polygons"].cpu()
        p_lbls = pred["labels"].cpu().numpy().astype(int)
        p_scores = pred["scores"].cpu().numpy()
        p_angs = pred["boxes"][:, 4].cpu().numpy()

        for poly, lbl, sc, ang in zip(p_polys, p_lbls, p_scores, p_angs):
            pts = poly.view(4, 2).numpy()
            ax.add_patch(
                MplPolygon(
                    pts,
                    closed=True,
                    fill=False,
                    edgecolor="#33AA33",
                    lw=1.5,
                    linestyle="--",
                )
            )
            ax.plot(
                [pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]], color="#FF8800", lw=1.5
            )
            ax.text(
                pts[:, 0].mean(),
                pts[:, 1].mean(),
                f"{labels_map[int(lbl)]} {math.degrees(float(ang)):.0f}°\n{sc:.2f}",
                color="#33AA33",
                fontsize=5,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#33AA33", lw=0.5),
            )

    for ax in axes[len(samples) :]:
        ax.axis("off")

    plt.tight_layout(pad=1.0)

    return {"pr_figure": fig_pr, "confusion_figure": fig_cm, "grid_figure": fig_grid}
