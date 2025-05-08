import os
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
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
    output_dir: str,
    device: torch.device,
    labels_map: dict,
    scale_factors: list,
    ratio_factors: list,
    obb_stats_by_size: dict,
    conf_thres: float = 0.25,
    iou_thres: float = 0.5,
    grid_shape: tuple = (3, 3),
) -> dict:
    """
    Full evaluation of a pretrained RetinaBabyFace on a test set.
    Returns various plots and metrics, including:
    - Precision–Recall curves (per class) + mAP
    - Confusion matrix with background (-1)
    - Boxplots: IoU and Angle MAE per class
    - F1-score vs Confidence threshold (per class)
    - Qualitative grid with predictions and GTs
    """

    # --- Load model ---
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()

    # --- Generate anchors ---
    resize_size = get_resize_size(test_loader)
    base_size, base_ratio = get_base_obb_stats(resize_size, obb_stats_by_size)
    anchors_xy, anchors_xywhr = generate_anchors_for_training(
        model, resize_size, device, base_size, base_ratio, scale_factors, ratio_factors
    )

    # --- Initialize accumulators ---
    per_class_true = {c: [] for c in labels_map}
    per_class_score = {c: [] for c in labels_map}
    iou_vals_per_class = {c: [] for c in labels_map}
    angle_errs_per_class = {c: [] for c in labels_map}
    stats_per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in labels_map}
    y_pred_cls, y_true_cls = [], []
    all_scores, all_pred_labels, all_gt_labels = [], [], []
    samples = []
    dataset = test_loader.dataset
    sample_idx = 0

    # --- Inference ---
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
                gt_polys = targets["boxes"][b][valid]
                gt_angs = targets["angles"][b][valid].reshape(-1)
                gt_lbls = targets["class_idx"][b][valid]

                gt_xywhr = xyxyxyxy2xywhr(
                    gt_polys, gt_angs.unsqueeze(-1), resize_size
                ).to(device)

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

                pred = outs[b]
                p_xywhr, p_polys = pred["boxes"].to(device), pred["polygons"].to(device)
                p_scores, p_lbls = pred["scores"].to(device), pred["labels"].to(device)
                G, M = gt_xywhr.size(0), p_xywhr.size(0)

                iou_matrix = (
                    batch_probiou(gt_xywhr, p_xywhr)
                    if G > 0 and M > 0
                    else torch.zeros(G, M, device=device)
                )
                matched_pred = torch.zeros(M, dtype=torch.bool, device=device)

                for i in range(G):
                    best_iou, j = iou_matrix[i].max(0)
                    gt_cls = int(gt_lbls[i].item())
                    matched = best_iou >= iou_thres

                    if matched:
                        stats_per_class[gt_cls]["tp"] += 1
                        iou_vals_per_class[gt_cls].append(best_iou.item())
                        pred_ang = p_xywhr[j, 4]
                        err_deg = abs((pred_ang - gt_angs[i]) * 180.0 / math.pi)
                        angle_errs_per_class[gt_cls].append(err_deg.item())
                        matched_pred[j] = True
                    else:
                        stats_per_class[gt_cls]["fn"] += 1

                    for c in labels_map:
                        per_class_true[c].append(int(gt_cls == c))
                        per_class_score[c].append(
                            float(p_scores[j].item()) if matched else 0.0
                        )

                    y_pred_cls.append(int(p_lbls[j].item()) if matched else -1)
                    y_true_cls.append(gt_cls)
                    all_gt_labels.append(gt_cls)
                    all_scores.append(float(p_scores[j].item()))
                    all_pred_labels.append(int(p_lbls[j].item()) if matched else -1)

                for k in range(M):
                    if not matched_pred[k]:
                        stats_per_class[int(p_lbls[k].item())]["fp"] += 1

    # --- Metrics: mAP & PR curves ---
    APs = {}
    for c in labels_map:
        y_t = np.array(per_class_true[c])
        y_s = np.array(per_class_score[c])
        APs[c] = (
            np.trapz(*precision_recall_curve(y_t, y_s)[0:2]) if y_t.sum() > 0 else 0.0
        )
    map_global = np.mean(list(APs.values()))

    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    for c, name in labels_map.items():
        y_t = np.array(per_class_true[c])
        y_s = np.array(per_class_score[c])
        if y_t.sum() == 0:
            ax_pr.step([0, 1], [1, 1], where="post", label=f"{name} (npos=0)")
        else:
            prec, rec, _ = precision_recall_curve(y_t, y_s)
            ax_pr.step(rec, prec, where="post", label=f"{name} AP={APs[c]:.3f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision–Recall (mAP={map_global:.3f})")
    ax_pr.legend(loc="upper right", fontsize=8)
    fig_pr.tight_layout()

    # --- Confusion Matrix ---
    cm_labels = list(labels_map) + [-1]
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=cm_labels)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        cm, display_labels=[labels_map.get(c, "BG") for c in cm_labels]
    )
    disp.plot(
        ax=ax_cm, cmap="Blues", colorbar=False, xticks_rotation=45, values_format="d"
    )
    ax_cm.set_title("Confusion Matrix", pad=20)
    fig_cm.tight_layout()

    # --- Boxplots: IoU & Angle Error ---
    df_iou = pd.DataFrame(
        [
            {"class": labels_map[c], "iou": v}
            for c, vals in iou_vals_per_class.items()
            for v in vals
        ]
    )
    df_ang = pd.DataFrame(
        [
            {"class": labels_map[c], "error°": v}
            for c, vals in angle_errs_per_class.items()
            for v in vals
        ]
    )

    fig_box_iou, ax_box_iou = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=df_iou,
        x="class",
        y="iou",
        hue="class",
        legend=False,
        notch=True,
        palette="pastel",
        ax=ax_box_iou,
    )
    sns.stripplot(
        data=df_iou,
        x="class",
        y="iou",
        color="gray",
        size=3,
        jitter=True,
        ax=ax_box_iou,
    )
    ax_box_iou.set_title("IoU Distribution per Class")
    ax_box_iou.set_xlabel("")
    ax_box_iou.set_ylabel("IoU")
    fig_box_iou.tight_layout()

    fig_box_ang, ax_box_ang = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=df_ang,
        x="class",
        y="error°",
        hue="class",
        legend=False,
        notch=True,
        palette="pastel",
        ax=ax_box_ang,
    )
    sns.stripplot(
        data=df_ang,
        x="class",
        y="error°",
        color="gray",
        size=3,
        jitter=True,
        ax=ax_box_ang,
    )
    ax_box_ang.set_title("Angle‐Error Distribution per Class")
    ax_box_ang.set_xlabel("")
    ax_box_ang.set_ylabel("Error (°)")
    fig_box_ang.tight_layout()

    # --- F1 Score vs Confidence Threshold ---
    thresholds = np.linspace(0, 1, 50)
    f1_matrix = np.zeros((len(thresholds), len(labels_map)))
    y_true_arr = np.array(all_gt_labels)

    for i, t in enumerate(thresholds):
        y_pred_t = [
            lab if sc >= t else -1 for sc, lab in zip(all_scores, all_pred_labels)
        ]
        f1s = f1_score(
            y_true_arr, y_pred_t, labels=list(labels_map), average=None, zero_division=0
        )
        f1_matrix[i] = f1s

    default_idx = np.argmin(np.abs(thresholds - conf_thres))
    f1_at_def = f1_matrix[default_idx]

    fig_f1curve, ax_f1curve = plt.subplots(figsize=(6, 4))
    for j, c in enumerate(labels_map):
        label = labels_map[c]
        ax_f1curve.step(
            thresholds,
            f1_matrix[:, j],
            where="post",
            label=f"{label} (F1@{conf_thres:.2f}={f1_at_def[j]:.3f})",
        )
    ax_f1curve.set_xlabel("Confidence Threshold")
    ax_f1curve.set_ylabel("F1 Score")
    ax_f1curve.set_title("F1 vs. Confidence Threshold")
    ax_f1curve.legend(fontsize=7, loc="upper right", ncol=2)
    fig_f1curve.tight_layout()

    # 10) Qualitative grid
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

        # GT in blue + red edge
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

        # preds in green + orange edge
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

    save_individual_predictions(samples, labels_map, output_dir)

    return {
        "pr_figure": fig_pr,
        "confusion_figure": fig_cm,
        "iou_boxplot_figure": fig_box_iou,
        "angle_boxplot_figure": fig_box_ang,
        "f1_threshold_figure": fig_f1curve,
        "grid_figure": fig_grid,
        "mAP": map_global,
    }


def save_individual_predictions(samples, labels_map, output_dir):
    """
    Save individual test images with both ground truth and predicted bounding boxes.

    Each image will be annotated with:
    - Ground truth (blue polygon with red orientation line and class + angle)
    - Predictions (green dashed polygon with orange orientation and class + angle + score)

    Args:
        samples (List[Tuple]): List of tuples, each containing:
            (image_tensor, prediction_dict, filename, gt_polygons, gt_angles, gt_labels)
        labels_map (Dict[int, str]): Mapping from class indices to human-readable labels
        output_dir (str): Directory where annotated images will be saved
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for img_t, pred, fname, gt_polys, gt_angs, gt_lbls in samples:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(denormalize_image(img_t))
        ax.axis("off")
        ax.set_aspect("equal")

        # Draw ground truth boxes in blue with red edge direction
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

        # Draw predicted boxes in green dashed lines with orange edge direction
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

        # Save figure with the same filename in the output directory
        save_path = os.path.join(output_dir, os.path.basename(fname))
        fig.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    print(f"[INFO] Saved individual predictions to {output_dir}")
