import os
import math
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects as pe
from matplotlib.patches import Polygon as MplPolygon
from torch.utils.data import DataLoader
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
# I. Checkpoint & Anchor Setup
# -----------------------------------------------------------------------------


def load_model_checkpoint(model: torch.nn.Module, path: str, device: torch.device):
    """Load weights from a checkpoint into `model` and set eval mode."""
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    logging.info(f"Checkpoint loaded from {path}")


def prepare_anchors(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    scale_factors: List[float],
    ratio_factors: List[float],
    obb_stats: Dict[Tuple[int, int], Dict[str, float]],
):
    """Compute input resize, then generate and return anchors."""
    resize_size = get_resize_size(loader)  # (W, H)
    base_size, base_ratio = get_base_obb_stats(resize_size, obb_stats)
    anchors_xy, anchors_xywhr = generate_anchors_for_training(
        model, resize_size, device, base_size, base_ratio, scale_factors, ratio_factors
    )
    logging.info(f"Generated {anchors_xy.shape[0]} anchors")
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
    Run model inference + rotated NMS on the test_loader and accumulate:
      - per‐class true/score for PR
      - IoU & angle errors for boxplots
      - confusion/f1 statistics
      - raw lists of scores & labels for global F1 curve
      - a small set of samples for qualitative grid
    """

    per_class_true  = {c: [] for c in labels_map}
    per_class_score = {c: [] for c in labels_map}
    iou_errs        = {c: [] for c in labels_map}
    angle_errs      = {c: [] for c in labels_map}
    y_true_cls, y_pred_cls = [], []
    all_scores, all_pred_labels, all_gt_labels = [], [], []
    stats_per_class = {c: {"tp":0,"fp":0,"fn":0} for c in labels_map}
    samples = []
    dataset = loader.dataset
    idx_ptr = 0

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Infer"):
            # 1) Lleva sólo las imágenes a GPU
            imgs    = batch["image"].to(device)
            targets = batch["target"]

            # 2) Forward + NMS
            outs = infer_with_rotated_nms(
                model, imgs, anchors_xy, resize_size, conf_thres, iou_thres
            )
            B = imgs.size(0)

            # 3) Procesa cada muestra
            for b in range(B):
                base = dataset.file_list[idx_ptr]
                idx_ptr += 1

                # Extrae ground-truth
                valid       = targets["valid_mask"][b]
                gt_boxes    = targets["boxes"][b][valid]
                gt_angles   = targets["angles"][b][valid].view(-1)
                gt_labels   = targets["class_idx"][b][valid]

                # Calcula IoU entre gt y preds
                gt_xywhr = xyxyxyxy2xywhr(
                    gt_boxes, gt_angles.unsqueeze(-1), resize_size
                ).to(device)

                # ---- GUARDAR EN SAMPLES (siempre CPU!) ----
                img_cpu       = imgs[b].cpu()
                out_cpu       = {k: v.cpu().detach() for k,v in outs[b].items()}
                gt_boxes_cpu  = gt_boxes.cpu()
                gt_angles_cpu = gt_angles.cpu()
                gt_labels_cpu = gt_labels.cpu()
                samples.append(
                    (img_cpu, out_cpu, base, gt_boxes_cpu, gt_angles_cpu, gt_labels_cpu)
                )
                # -------------------------------------------

                # Prepara predicciones en GPU para matching
                pred_boxes  = outs[b]["boxes"].to(device)
                pred_polys  = outs[b]["polygons"].to(device)
                pred_scores = outs[b]["scores"].to(device)
                pred_lbls   = outs[b]["labels"].to(device)

                G, M = gt_xywhr.size(0), pred_boxes.size(0)
                iou_matrix = (
                    batch_probiou(gt_xywhr, pred_boxes)
                    if G and M else torch.zeros(G, M, device=device)
                )
                matched = torch.zeros(M, dtype=torch.bool, device=device)

                # Match GT → preds
                for i in range(G):
                    cls = int(gt_labels[i].item())
                    if M == 0:
                        stats_per_class[cls]["fn"] += 1
                        for c in labels_map:
                            per_class_true[c].append(int(c == cls))
                            per_class_score[c].append(0.0)
                        y_true_cls.append(cls)
                        y_pred_cls.append(-1)
                        all_gt_labels.append(cls)
                        all_scores.append(0.0)
                        all_pred_labels.append(-1)
                        continue

                    best_iou, j = iou_matrix[i].max(0)
                    is_pos = best_iou >= iou_thres
                    if is_pos:
                        stats_per_class[cls]["tp"] += 1
                        iou_errs[cls].append(best_iou.item())
                        err_deg = abs((pred_boxes[j,4] - gt_angles[i]) * 180/math.pi)
                        angle_errs[cls].append(err_deg.item())
                        matched[j] = True
                    else:
                        stats_per_class[cls]["fn"] += 1

                    for c in labels_map:
                        per_class_true[c].append(int(c == cls))
                        per_class_score[c].append(
                            pred_scores[j].item() if is_pos else 0.0
                        )
                    y_true_cls.append(cls)
                    y_pred_cls.append(int(pred_lbls[j].item()) if is_pos else -1)
                    all_gt_labels.append(cls)
                    all_scores.append(pred_scores[j].item() if is_pos else 0.0)
                    all_pred_labels.append(
                        int(pred_lbls[j].item()) if is_pos else -1
                    )

                # False positives
                for k in range(M):
                    if not matched[k]:
                        cls = int(pred_lbls[k].item())
                        stats_per_class[cls]["fp"] += 1

            # 4) Libera memoria GPU antes del siguiente batch
            del imgs, outs, targets
            torch.cuda.empty_cache()

    return {
        "per_class_true": per_class_true,
        "per_class_score": per_class_score,
        "iou_errs":       iou_errs,
        "angle_errs":     angle_errs,
        "stats_per_class":stats_per_class,
        "y_true_cls":     y_true_cls,
        "y_pred_cls":     y_pred_cls,
        "all_scores":     all_scores,
        "all_pred_labels":all_pred_labels,
        "all_gt_labels":  all_gt_labels,
        "samples":        samples,
    }


# -----------------------------------------------------------------------------
# III. Metric Computation & Plotting
# -----------------------------------------------------------------------------



def compute_map_and_pr(per_true, per_score):
    """Compute per‐class AP and global mAP."""
    APs = {}
    for c, y_t in per_true.items():
        y_s = np.array(per_score[c])
        APs[c] = average_precision_score(y_t, y_s) if sum(y_t) > 0 else 0.0
    return float(np.mean(list(APs.values()))), APs


def plot_precision_recall(per_true, per_score, labels_map, APs, mAP):
    """Matplotlib per‐class Precision–Recall curves with fixed color map."""
    # fixed class order & colors
    classes = list(labels_map.keys())
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6, 5))
    for idx, cls in enumerate(classes):
        name = labels_map[cls]
        y_t = np.array(per_true[cls])
        y_s = np.array(per_score[cls])
        color = cmap(idx)
        if y_t.sum() == 0:
            ax.step([0, 1], [1, 1], where="post",
                    color=color, linestyle="--",
                    label=f"{name} (npos=0)")
        else:
            prec, rec, _ = precision_recall_curve(y_t, y_s)
            ax.step(rec, prec, where="post", color=color,
                    label=f"{name} AP={APs[cls]:.3f}")
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"Precision–Recall (mAP={mAP:.3f})",
    )
    # legend outside, no frame
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=8, title="Classes")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout(rect=(0, 0, 0.85, 1))  # leave room for legend
    return fig


def plot_confusion_matrix(y_true, y_pred, labels_map):
    """Simple Matplotlib confusion matrix with annotated counts."""
    cm_labels = list(labels_map.keys()) + [-1]
    names = [labels_map.get(c, "BG") for c in cm_labels]
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
    support = cm.sum(axis=1)
    correct = np.diag(cm)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{correct[i]}/{support[i]}" if i==j else str(cm[i,j])
            ax.text(j, i, txt, ha="center", va="center", color="white" if cm[i,j]>cm.max()/2 else "black")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_boxplots(data, x_field, y_field, title, y_lim=None):
    """
    Generic Matplotlib boxplot + jittered scatter.
    `data` is a list of dicts with keys x_field, y_field.
    """
    # collect per‐category
    cats = sorted({d[x_field] for d in data})
    values = [[d[y_field] for d in data if d[x_field]==cat] for cat in cats]
    fig, ax = plt.subplots(figsize=(6, 4))
    # boxplot
    bp = ax.boxplot(values, labels=cats, notch=True, patch_artist=True,
                    boxprops=dict(facecolor="#DDDDDD", edgecolor="black"))
    # scatter jitter
    for i, vals in enumerate(values):
        x = np.random.normal(i+1, 0.08, size=len(vals))
        ax.scatter(x, vals, color="gray", s=6, alpha=0.6)
    ax.set_title(title)
    ax.set_ylabel(y_field)
    ax.set_xlabel("")
    if y_lim:
        ax.set_ylim(y_lim)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Helper: plot F1 vs confidence threshold, marking best F1 with red dashed line
# -----------------------------------------------------------------------------
def plot_f1_vs_threshold(all_gt, all_scores, all_pred, labels_map, default_th=0.25,
                         n_steps=100, th_min=0.05, th_max=0.95):
    """Matplotlib F1 vs. confidence threshold, marking each class’s best."""
    thresholds = np.linspace(th_min, th_max, n_steps)
    y_true = np.array(all_gt)
    classes = list(labels_map.keys())
    cmap = plt.get_cmap("tab10")

    # compute f1
    f1_mat = np.zeros((n_steps, len(classes)))
    for i, t in enumerate(thresholds):
        y_pred = [lbl if sc>=t else -1 for sc, lbl in zip(all_scores, all_pred)]
        f1s = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
        f1_mat[i,:] = f1s

    fig, ax = plt.subplots(figsize=(6, 4))
    for j, cls in enumerate(classes):
        vals = f1_mat[:,j]
        color = cmap(j)
        ax.plot(thresholds, vals, label=f"{labels_map[cls]} (F1@{default_th:.2f}={vals[np.abs(thresholds-default_th).argmin()]:.3f})",
                color=color, linewidth=1.5)
        # best point
        bi = vals.argmax()
        tb, fb = thresholds[bi], vals[bi]
        ax.axvline(tb, linestyle="--", color=color, linewidth=1)
        ax.scatter(tb, fb, color=color, s=30, zorder=3)
    ax.set_xlim(th_min, th_max)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 vs. Confidence Threshold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# IV. Qualitative Grid
# -----------------------------------------------------------------------------


def plot_qualitative_grid(
    samples, labels_map, grid_shape, mean, std
):
    """
    Plot a grid of images with GT boxes (blue/red) and predictions (green/orange),
    annotated with class, angle and score. No opaque text boxes.
    """
    rows, cols = grid_shape
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4), facecolor="white")
    axes = axes.flatten()

    for ax, (img_t, out, fname, gt_boxes, gt_ang, gt_lbl) in zip(axes, samples[:rows*cols]):
        # show
        ax.imshow(denormalize_image(img_t, mean=mean, std=std))
        ax.axis("off")
        ax.set_title(os.path.basename(fname), fontsize=8)

        # GT
        for pts, ang, cls in zip(gt_boxes, gt_ang, gt_lbl):
            pts_np = pts.view(4,2).numpy()
            poly = patches.Polygon(pts_np, closed=True, fill=False, edgecolor="#0055FF", linewidth=2)
            ax.add_patch(poly)
            # orientation
            ax.plot(pts_np[[0,1],0], pts_np[[0,1],1], color="#FF3333", linewidth=2)
            # label+angle
            lbl = f"{labels_map[int(cls)]}: {math.degrees(ang):.1f}°"
            cen = pts_np.mean(axis=0)
            ax.text(cen[0], cen[1], lbl, color="#0055FF", fontsize=6, fontweight="bold",
                    ha="center", va="center",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])

        # Predictions
        boxes_xywhr = out.get("boxes", None)
        for i, (poly_pts, cls, score) in enumerate(zip(out["polygons"], out["labels"], out["scores"])):
            pts_np = poly_pts.view(4,2).numpy()
            poly = patches.Polygon(pts_np, closed=True, fill=False,
                                   edgecolor="#33AA33", linewidth=1.5, linestyle="--")
            ax.add_patch(poly)
            ax.plot(pts_np[[0,1],0], pts_np[[0,1],1], color="#FF8800", linewidth=1.5)
            if boxes_xywhr is not None:
                ang_pred = math.degrees(float(boxes_xywhr[i,4]))
                lbl = f"{labels_map[int(cls)]}: {ang_pred:.1f}° / {score:.2f}"
            else:
                lbl = f"{labels_map[int(cls)]}: {score:.2f}"
            cen = pts_np.mean(axis=0)
            ax.text(cen[0], cen[1], lbl, color="#33AA33", fontsize=5,
                    ha="center", va="center",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # blank extra axes
    for ax in axes[len(samples):]:
        ax.axis("off")

    fig.tight_layout(pad=0.5)
    return fig


# -----------------------------------------------------------------------------
# V. Main entry
# -----------------------------------------------------------------------------


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
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> dict:
    # ensure output dir exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # I. Load checkpoint
    load_model_checkpoint(model, checkpoint_path, device)

    # II. Generate anchors
    resize_size, anchors_xy, anchors_xywhr = prepare_anchors(
        model, test_loader, device, scale_factors, ratio_factors, obb_stats_by_size
    )

    # III. Run inference + accumulate
    results = run_inference(
        model,
        test_loader,
        anchors_xy,
        resize_size,
        conf_thres,
        iou_thres,
        device,
        labels_map,
    )

    # IV. Compute mAP & PR
    mAP, APs = compute_map_and_pr(results["per_class_true"], results["per_class_score"])
    fig_pr = plot_precision_recall(
        results["per_class_true"], results["per_class_score"], labels_map, APs, mAP
    )

    # V. Confusion
    fig_cm = plot_confusion_matrix(
        results["y_true_cls"], results["y_pred_cls"], labels_map
    )

    # VI. Boxplots
    iou_data = [
        {"class": labels_map[c], "iou": v}
        for c, vals in results["iou_errs"].items()
        for v in vals
    ]
    ang_data = [
        {"class": labels_map[c], "error°": v}
        for c, vals in results["angle_errs"].items()
        for v in vals
    ]
    fig_box_iou = plot_boxplots(iou_data, "class", "iou", "IoU Distribution per Class")
    fig_box_ang = plot_boxplots(
        ang_data, "class", "error°", "Angle‐Error Distribution per Class"
    )

    # VII. F1 vs threshold
    fig_f1 = plot_f1_vs_threshold(
        results["all_gt_labels"],
        results["all_scores"],
        results["all_pred_labels"],
        labels_map,
        default_th=conf_thres,
    )

    # VIII. Qualitative grid
    fig_grid = plot_qualitative_grid(
        results["samples"], labels_map, grid_shape, mean, std
    )

    # IX. Save individual images
    save_individual_predictions(results["samples"], labels_map, output_dir, mean, std)

    return {
        "pr_figure": fig_pr,
        "confusion_figure": fig_cm,
        "iou_boxplot_figure": fig_box_iou,
        "angle_boxplot_figure": fig_box_ang,
        "f1_threshold_figure": fig_f1,
        "grid_figure": fig_grid,
        "mAP": mAP,
    }


def save_individual_predictions(samples, labels_map, output_dir, mean, std):
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
        mean (Tuple[float, float, float]): Mean values for denormalization
        std (Tuple[float, float, float]): Standard deviation values for denormalization
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for img_t, pred, fname, gt_polys, gt_angs, gt_lbls in samples:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(denormalize_image(img_t, mean=mean, std=std))
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
