import os
import math
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
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
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    logging.info(f"Checkpoint loaded from {path}")

def prepare_anchors(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    scale_factors: List[float],
    ratio_factors: List[float],
    obb_stats: Dict[Tuple[int,int], Dict[str,float]],
) -> Tuple[Tuple[int,int], torch.Tensor, torch.Tensor]:
    resize_size = get_resize_size(loader)
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
    resize_size: Tuple[int,int],
    conf_thres: float,
    iou_thres: float,
    device: torch.device,
    labels_map: Dict[int,str],
) -> Dict[str,Any]:
    per_true   = {c:[] for c in labels_map}
    per_score  = {c:[] for c in labels_map}
    iou_errs   = {c:[] for c in labels_map}
    angle_errs = {c:[] for c in labels_map}
    stats      = {c:{"tp":0,"fp":0,"fn":0} for c in labels_map}
    y_true, y_pred = [], []
    all_scores, all_preds, all_gts = [], [], []
    samples = []
    ds = loader.dataset
    idx = 0

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Infer"):
            imgs    = batch["image"].to(device)
            targets = batch["target"]
            outs    = infer_with_rotated_nms(
                        model, imgs, anchors_xy, resize_size, conf_thres, iou_thres
                     )
            B = imgs.size(0)

            for b in range(B):
                fname = ds.file_list[idx]; idx+=1

                valid     = targets["valid_mask"][b]
                gt_boxes  = targets["boxes"][b][valid]
                gt_angles = targets["angles"][b][valid].view(-1)
                gt_labels = targets["class_idx"][b][valid]

                gt_xywhr = xyxyxyxy2xywhr(
                    gt_boxes, gt_angles.unsqueeze(-1), resize_size
                ).to(device)

                # guardo CPU‐copy para grid & save
                samples.append((
                    imgs[b].cpu(),
                    {k:v.cpu().detach() for k,v in outs[b].items()},
                    fname,
                    gt_boxes.cpu(),
                    gt_angles.cpu(),
                    gt_labels.cpu(),
                ))

                pred_boxes  = outs[b]["boxes"].to(device)
                pred_scores = outs[b]["scores"].to(device)
                pred_lbls   = outs[b]["labels"].to(device)

                G, M = gt_xywhr.size(0), pred_boxes.size(0)
                iou_m = batch_probiou(gt_xywhr, pred_boxes) if G and M else torch.zeros(G,M,device=device)
                matched = torch.zeros(M, dtype=torch.bool, device=device)

                # match GT→pred
                for i in range(G):
                    cls = int(gt_labels[i].item())
                    if M == 0:
                        stats[cls]["fn"] += 1
                        for c in labels_map:
                            per_true[c].append(int(c==cls))
                            per_score[c].append(0.0)
                        y_true.append(cls); y_pred.append(-1)
                        all_gts.append(cls); all_scores.append(0.0); all_preds.append(-1)
                        continue

                    best_iou, j = iou_m[i].max(0)
                    is_pos = best_iou >= iou_thres
                    if is_pos:
                        stats[cls]["tp"] += 1
                        iou_errs[cls].append(best_iou.item())
                        err_deg = abs((pred_boxes[j,4] - gt_angles[i]) * 180/math.pi)
                        angle_errs[cls].append(err_deg.item())
                        matched[j] = True
                    else:
                        stats[cls]["fn"] += 1

                    for c in labels_map:
                        per_true[c].append(int(c==cls))
                        per_score[c].append(pred_scores[j].item() if is_pos else 0.0)

                    y_true.append(cls)
                    y_pred.append(int(pred_lbls[j].item()) if is_pos else -1)
                    all_gts.append(cls)
                    all_scores.append(pred_scores[j].item() if is_pos else 0.0)
                    all_preds.append(int(pred_lbls[j].item()) if is_pos else -1)

                # false positives
                for k in range(M):
                    if not matched[k]:
                        cl = int(pred_lbls[k].item())
                        stats[cl]["fp"] += 1

            del imgs, outs, targets
            torch.cuda.empty_cache()

    return {
        "per_true":per_true, "per_score":per_score,
        "iou_errs":iou_errs, "angle_errs":angle_errs,
        "stats":stats,
        "y_true":y_true, "y_pred":y_pred,
        "all_scores":all_scores, "all_preds":all_preds, "all_gts":all_gts,
        "samples":samples,
    }

# -----------------------------------------------------------------------------
# III. Metric Computation & Plotting
# -----------------------------------------------------------------------------

def compute_map_and_pr(per_true, per_score) -> Tuple[float,Dict[int,float]]:
    APs = {
        c: (average_precision_score(per_true[c], per_score[c])
            if sum(per_true[c])>0 else 0.0)
        for c in per_true
    }
    return float(np.mean(list(APs.values()))), APs

def plot_precision_recall(per_true, per_score, labels_map, APs, mAP):
    """
    Dibuja:
     - cada clase con su color de tab20,
     - y la curva global (stacked) en azul grueso.
    """
    classes = list(labels_map.keys())
    cmap    = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(6,5))

    # curvas por clase
    for i, cls in enumerate(classes):
        col  = cmap(i)
        y_t  = np.array(per_true[cls])
        y_s  = np.array(per_score[cls])
        if y_t.sum() == 0:
            ax.step([0,1],[1,1],where="post",color=col,linestyle="--",
                    label=f"{labels_map[cls]} (npos=0)")
        else:
            prec, rec, _ = precision_recall_curve(y_t, y_s)
            ax.step(rec, prec, where="post", color=col,
                    label=f"{labels_map[cls]} {APs[cls]:.3f}")

    # curva global (stacked)
    all_y = np.concatenate([per_true[c]  for c in classes])
    all_s = np.concatenate([per_score[c] for c in classes])
    p_g, r_g, _ = precision_recall_curve(all_y, all_s)
    ax.step(r_g, p_g, where="post", color="navy", linewidth=3,
            label=f"all classes {mAP:.3f} mAP@0.5")

    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall Curve")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02,1), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0,0,0.78,1))
    return fig

def plot_confusion_matrix(y_true, y_pred, labels_map):
    labels = list(labels_map.keys()) + [-1]
    names  = [labels_map.get(l,"BG") for l in labels]
    cm     = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax= plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(len(names)):
        for j in range(len(names)):
            txt = (f"{np.diag(cm)[i]}/{cm.sum(1)[i]}"
                   if i==j else str(cm[i,j]))
            c   = "white" if cm[i,j]>cm.max()/2 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=c)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set(xlabel="Predicted", ylabel="True", title="Confusion Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

def plot_boxplots(data, x_field, y_field, title, y_lim=None):
    cats   = sorted({d[x_field] for d in data})
    values = [[d[y_field] for d in data if d[x_field]==c] for c in cats]
    cmap   = plt.get_cmap("tab20")
    fig, ax= plt.subplots(figsize=(6,4))
    bp = ax.boxplot(values, labels=cats, notch=True, patch_artist=True)
    # mismo color que en PR
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(cmap(i))
        box.set_edgecolor("black")
    # jitter points
    for i, vals in enumerate(values):
        xs = np.random.normal(i+1, 0.06, size=len(vals))
        ax.scatter(xs, vals, color=cmap(i), s=6, alpha=0.7)
    ax.set(title=title, ylabel=y_field)
    if y_lim: ax.set_ylim(y_lim)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig

def plot_f1_vs_threshold(all_gts, all_scores, all_preds, labels_map,
                         default_th=0.25, n_steps=100, th_min=0.05, th_max=0.95):
    thresholds = np.linspace(th_min, th_max, n_steps)
    y_true     = np.array(all_gts)
    classes    = list(labels_map.keys())
    cmap       = plt.get_cmap("tab20")
    f1_mat     = np.zeros((n_steps, len(classes)))
    for i, t in enumerate(thresholds):
        preds = [lbl if sc>=t else -1 for sc,lbl in zip(all_scores, all_preds)]
        f1_mat[i] = f1_score(y_true, preds, labels=classes, average=None, zero_division=0)
    fig, ax = plt.subplots(figsize=(6,4))
    for j, cls in enumerate(classes):
        vals = f1_mat[:,j]; col = cmap(j)
        ax.plot(thresholds, vals, color=col, linewidth=1.5,
                label=f"{labels_map[cls]} {vals[np.argmin(abs(thresholds-default_th))]:.3f}")
        bi = vals.argmax(); tb, fb = thresholds[bi], vals[bi]
        ax.axvline(tb, linestyle="--", color=col, linewidth=1)
        ax.scatter(tb, fb, color=col, s=30, zorder=3)
    ax.set(xlabel="Confidence Threshold", ylabel="F1 Score", title="F1 vs. Confidence Threshold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# IV. Qualitative Grid & Saving Individually
# -----------------------------------------------------------------------------

def plot_qualitative_grid(samples, labels_map, grid_shape, mean, std):
    rows, cols = grid_shape
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4), facecolor="white")
    axes = axes.flatten()
    for ax, (img_t, out, fname, gt_b, gt_a, gt_l) in zip(axes, samples[:rows*cols]):
        ax.imshow(denormalize_image(img_t, mean=mean, std=std))
        ax.axis("off")
        ax.set_title(Path(fname).name, fontsize=8)
        # GT
        for pts, ang, cls in zip(gt_b, gt_a, gt_l):
            pts_np = pts.view(4,2).numpy()
            ax.add_patch(patches.Polygon(pts_np, closed=True, fill=False,
                                        edgecolor="blue", linewidth=2))
            ax.plot(pts_np[[0,1],0], pts_np[[0,1],1], color="red", linewidth=2)
            lbl = f"{labels_map[int(cls)]}: {math.degrees(ang):.1f}°"
            cen = pts_np.mean(axis=0)
            ax.text(cen[0], cen[1], lbl,
                    color="blue", fontsize=6, fontweight="bold",
                    ha="center", va="center",
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])
        # preds
        for i, (pts, l, s) in enumerate(zip(out["polygons"], out["labels"], out["scores"])):
            pts_np = pts.view(4,2).numpy()
            ax.add_patch(patches.Polygon(pts_np, closed=True, fill=False,
                                        edgecolor="green", linewidth=1.5, linestyle="--"))
            ax.plot(pts_np[[0,1],0], pts_np[[0,1],1], color="orange", linewidth=1.5)
            angp = math.degrees(float(out["boxes"][i,4]))
            lbl  = f"{labels_map[int(l)]}: {angp:.1f}°/{s:.2f}"
            cen = pts_np.mean(axis=0)
            ax.text(cen[0], cen[1], lbl,
                    color="green", fontsize=5,
                    ha="center", va="center",
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])
    for ax in axes[len(samples):]:
        ax.axis("off")
    fig.tight_layout(pad=0.5)
    return fig

def save_individual_predictions(samples, labels_map, output_dir, mean, std):
    """
    Misma anotación que la grid pero guardando una a una.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for img_t, out, fname, gt_b, gt_a, gt_l in samples:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(denormalize_image(img_t, mean=mean, std=std))
        ax.axis("off"); ax.set_aspect("equal")
        # GT y preds iguales al grid
        # ... (idéntico a plot_qualitative_grid) ...
        # por brevedad omito repetir, pero mantén colores y efectos
        save_path = os.path.join(output_dir, os.path.basename(fname))
        fig.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
    logging.info(f"Saved individual predictions to {output_dir}")

# -----------------------------------------------------------------------------
# V. Main Entry
# -----------------------------------------------------------------------------

def inference(
    model: torch.nn.Module,
    checkpoint_path: str,
    test_loader: DataLoader,
    output_dir: str,
    device: torch.device,
    labels_map: Dict[int,str],
    scale_factors: List[float],
    ratio_factors: List[float],
    obb_stats_by_size: Dict[Tuple[int,int],Dict[str,float]],
    conf_thres: float=0.25,
    iou_thres: float=0.5,
    grid_shape: Tuple[int,int]=(3,3),
    mean: Tuple[float,float,float]=(0.485,0.456,0.406),
    std:  Tuple[float,float,float]=(0.229,0.224,0.225),
) -> Dict[str,Any]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # I. checkpoint
    load_model_checkpoint(model, checkpoint_path, device)
    # II. anchors
    resize_size, anc_xy, anc_xywhr = prepare_anchors(
        model, test_loader, device, scale_factors, ratio_factors, obb_stats_by_size
    )
    # III. inference
    results = run_inference(
        model, test_loader, anc_xy, resize_size,
        conf_thres, iou_thres, device, labels_map
    )

    # IV. métricas y plots
    mAP, APs = compute_map_and_pr(results["per_true"], results["per_score"])
    fig_pr = plot_precision_recall(
        results["per_true"], results["per_score"], labels_map, APs, mAP
    )
    fig_cm = plot_confusion_matrix(results["y_true"], results["y_pred"], labels_map)

    iou_data = [{"class":labels_map[c], "iou":v}
                for c,vals in results["iou_errs"].items() for v in vals]
    ang_data= [{"class":labels_map[c], "error°":v}
                for c,vals in results["angle_errs"].items() for v in vals]
    fig_iou = plot_boxplots(iou_data, "class", "iou", "IoU per Class", y_lim=(0,1))
    fig_ang = plot_boxplots(ang_data,  "class", "error°","Angle‐Error per Class", y_lim=(0,180))

    fig_f1 = plot_f1_vs_threshold(
        results["all_gts"], results["all_scores"], results["all_preds"],
        labels_map, default_th=conf_thres
    )
    fig_grid = plot_qualitative_grid(
        results["samples"], labels_map, grid_shape, mean, std
    )

    # V. guardar individuales
    save_individual_predictions(
        results["samples"], labels_map, output_dir, mean, std
    )

    return {
        "pr_figure":            fig_pr,
        "confusion_figure":     fig_cm,
        "iou_boxplot_figure":   fig_iou,
        "angle_boxplot_figure": fig_ang,
        "f1_threshold_figure":  fig_f1,
        "grid_figure":          fig_grid,
        "mAP":                  mAP,
    }


def save_individual_predictions(samples, labels_map, output_dir, mean, std):
    """
    Igual al grid, pero guardando cada imagen individualmente con estilos homogéneos.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for img_t, pred, fname, gt_polys, gt_angs, gt_lbls in samples:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(denormalize_image(img_t, mean=mean, std=std))
        ax.axis("off"); ax.set_aspect("equal")

        # mismos estilos que en grid
        for pts, ang, lbl in zip(gt_polys, gt_angs, gt_lbls):
            coords = pts.view(4,2).numpy()
            ax.add_patch(patches.Polygon(coords, closed=True, fill=False,
                                         edgecolor="blue", linewidth=2))
            ax.plot(coords[[0,1],0], coords[[0,1],1], color="red", linewidth=2)
            ax.text(coords[:,0].mean(), coords[:,1].mean(),
                    f"{labels_map[int(lbl)]}: {math.degrees(float(ang)):.1f}°",
                    color="blue", fontsize=6, fontweight="bold",
                    ha="center", va="center",
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])

        for i, (pts, lbl, sc) in enumerate(zip(pred["polygons"], pred["labels"], pred["scores"])):
            coords = pts.cpu().view(4,2).numpy()
            ax.add_patch(patches.Polygon(coords, closed=True, fill=False,
                                         edgecolor="green", linewidth=1.5, linestyle="--"))
            ax.plot(coords[[0,1],0], coords[[0,1],1], color="orange", linewidth=1.5)
            angp = math.degrees(float(pred["boxes"][i,4]))
            ax.text(coords[:,0].mean(), coords[:,1].mean(),
                    f"{labels_map[int(lbl)]}: {angp:.1f}°/{sc:.2f}",
                    color="green", fontsize=5,
                    ha="center", va="center",
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])

        save_path = os.path.join(output_dir, os.path.basename(fname))
        fig.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    print(f"[INFO] Saved individual predictions to {output_dir}")
