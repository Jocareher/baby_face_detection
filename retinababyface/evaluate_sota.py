# evaluate_sota.py
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from benchmark.benchmark import (
    read_sota_preds_xywhr_xyxy,
    greedy_match,
    read_gt_baby_xywhr,
    read_yolo_oriented_preds_xywhr,
    read_pcn_preds_xywhr,
    count_adults_in_gt,
)
from engine.inference import (
    plot_precision_recall,
    compute_map_and_pr,
    plot_boxplots,
    plot_confusion_matrix,
    plot_f1_vs_threshold,
)
from data_setup.augmentations import wrap_to_pi

LABELS_MAP: Dict[int, str] = {
    0: "Leftside",
    1: "3/4 Leftside",
    2: "Frontal",
    3: "3/4 Rightside",
    4: "Rightside",
}


def evaluate_sota(
    data_root: Path,
    split: str,
    sota_dir: Path,
    out_dir: Path,
    iou_th: float = 0.5,
    min_score: float = 0.0,
    aabb_mode: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a SOTA (state-of-the-art) face detector against ground truth (GT) data for baby faces.
    The evaluation includes:
      - Precision-Recall (PR) and Average Precision (AP) for face/no-face classification.
      - Recall per orientation class (e.g., frontal, left-side, etc.).
      - Global false positives (FP) count, including unmatched predictions per image.
      - IoU analysis for true positives (TP), including histograms, boxplots, and statistics.
      - Exclusion of class -1 (non-baby/adult) from matching (all predictions on adults are treated as FP).

    Outputs:
      - PR curve for face/no-face classification.
      - Recall per orientation class as a bar plot.
      - IoU statistics and distribution plots.
      - CSV files for matches and summary metrics.

    Args:
        data_root (Path): Root directory of the dataset (contains split/images and split/labels).
        split (str): Dataset split to evaluate (e.g., "test").
        sota_dir (Path): Directory containing SOTA predictions in .txt format.
        out_dir (Path): Output directory for evaluation results.
        iou_th (float): IoU threshold for matching predictions to ground truth.
        min_score (float): Minimum score threshold for filtering predictions.
        aabb_mode (bool): If True, treat all boxes as axis-aligned bounding boxes (AABB).

    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics and statistics.
    """
    images_dir = data_root / split / "images"
    labels_dir = data_root / split / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== PR global (face/no-face) =====
    # Track true positives (TP) and false positives (FP) for face/no-face classification
        # PR global cara/no-cara
    per_true_face: Dict[int, List[int]]  = {0: []}
    per_score_face: Dict[int, List[float]] = {0: []}

    # Initialize dictionaries to track ground truth (GT), true positives (TP), and false negatives (FN) per class
    gt_per_cls = {c: 0 for c in LABELS_MAP}
    tp_per_cls = {c: 0 for c in LABELS_MAP}
    fn_per_cls = {c: 0 for c in LABELS_MAP}

    # Initialize a counter for global false positives (FP)
    fp_global = 0

    # Initialize lists and dictionaries to store IoU and angle error statistics for true positives (TPs)
    all_iou: List[float] = []  # List to store IoU values for all TPs
    all_scores_matched: List[float] = []  # List to store confidence scores for matched predictions
    iou_per_cls: Dict[int, List[float]] = {c: [] for c in LABELS_MAP}  # IoU values per class
    iou_data_for_boxplot: List[Dict[str, Any]] = []  # Data for IoU boxplots
    angle_errs_per_cls: Dict[int, List[float]] = {c: [] for c in LABELS_MAP}  # Angle errors per class
    angle_data_for_boxplot: List[Dict[str, Any]] = []  # Data for angle error boxplots

    # Helper function to get the size of an image
    def img_size(p: Path) -> Tuple[int, int]:
        with Image.open(p) as im:
            return im.size

    # Collect all image files with supported extensions
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    jpgs: List[Path] = []
    for pat in exts:
        jpgs += list(images_dir.glob(pat))
    jpgs = sorted(jpgs)

    # Iterate over all images in the dataset
    for img_p in jpgs:
        stem = img_p.stem
        gt_p = labels_dir / f"{stem}.txt"  # Path to ground truth file
        pr_p = sota_dir / f"{stem}.txt"  # Path to predictions file

        W, H = img_size(img_p)  # Get image dimensions

        # --- Process ground truth (GT) data ---
        gt_xywhr, gt_cls = read_gt_baby_xywhr(gt_p, (W, H))  # Read GT bounding boxes and classes
        if gt_cls.numel() > 0:
            # Exclude adults (-1) from GT data
            keep_baby = gt_cls != -1
            gt_xywhr_baby = gt_xywhr[keep_baby]
            gt_cls_baby = gt_cls[keep_baby]
        else:
            gt_xywhr_baby = gt_xywhr
            gt_cls_baby = gt_cls

        # Update GT counts per class
        for c in gt_cls_baby.tolist():
            gt_per_cls[int(c)] += 1

        # --- Process predictions ---
        if aabb_mode:
            # Read predictions as axis-aligned bounding boxes (AABB)
            pr_xywhr, pr_scores = read_sota_preds_xywhr_xyxy(
                pred_txt_path=pr_p, img_wh=(W, H), min_score=min_score
            )
        else:
            # Read predictions with orientation (e.g., PCN format)
            pr_xywhr, pr_scores = read_pcn_preds_xywhr(
                pred_txt_path=pr_p, img_wh=(W, H), min_score=min_score
            )

        # If no baby faces are present in GT, all predictions are false positives
        if gt_xywhr_baby.numel() == 0:
            fp_global += int(pr_xywhr.shape[0])  # Increment global FP count
            for s in pr_scores.tolist():
                per_true_face[0].append(0)  # No true positives
                per_score_face[0].append(float(s))  # Add prediction scores
            continue

        # Perform matching between GT and predictions based on IoU
        matches, unmatched_gt, unmatched_pr = greedy_match(
            gt_xywhr_baby, pr_xywhr, pr_scores, iou_th=iou_th
        )

        # Update precision-recall data for face/no-face classification
        matched_pr_idx = set([m for (_, m, _) in matches])
        for j, s in enumerate(pr_scores.tolist()):
            is_tp = 1 if j in matched_pr_idx else 0  # Check if prediction is a true positive
            per_true_face[0].append(is_tp)
            per_score_face[0].append(float(s))

        # Process matched predictions and compute IoU and angle errors
        rows_matches = []
        for gi, pj, iou in matches:
            c = int(gt_cls_baby[gi].item())  # Class of the matched GT
            class_name = LABELS_MAP[c]
            iou_val = float(iou)
            score_val = float(pr_scores[pj].item())

            # Update IoU statistics
            all_iou.append(iou_val)
            all_scores_matched.append(score_val)
            iou_per_cls[c].append(iou_val)
            iou_data_for_boxplot.append({"class": class_name, "IoU": iou_val})

            # Compute angle error if predictions include orientation
            if not aabb_mode:
                dtheta = wrap_to_pi(pr_xywhr[pj, 4] - gt_xywhr_baby[gi, 4])
                err_deg = float(torch.abs(dtheta) * 180.0 / math.pi)
                angle_errs_per_cls[c].append(err_deg)
                angle_data_for_boxplot.append({"class": class_name, "error°": err_deg})

            # Increment true positive count for the class
            tp_per_cls[c] += 1
            rows_matches.append(
                [img_p.name, int(gi), int(pj), score_val, iou_val, class_name]
            )

        # Update false negative counts for unmatched GT
        for gi in unmatched_gt:
            c = int(gt_cls_baby[gi].item())
            fn_per_cls[c] += 1

        # Update global false positive count for unmatched predictions
        fp_global += len(unmatched_pr)

    # ===== Compute Metrics & Generate Plots =====
    # Compute mAP and AP for face/no-face classification
    mAP, APs = compute_map_and_pr(per_true_face, per_score_face)
    ap_face = APs[0]

    # Generate precision-recall curve for face/no-face classification
    pr_fig = plot_precision_recall(per_true_face, per_score_face, labels_map={0: "Face"}, mAP=mAP)
    pr_fig.savefig(out_dir / "precision_recall_face.png", dpi=150, bbox_inches="tight")
    plt.close(pr_fig)

    # Compute recall per orientation class
    recalls = {c: (tp_per_cls[c] / gt_per_cls[c]) if gt_per_cls[c] > 0 else 0.0 for c in LABELS_MAP}

    # Generate bar plot for recall per orientation class
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = list(LABELS_MAP.keys())
    ax.bar([LABELS_MAP[x] for x in xs], [recalls[x] for x in xs])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title("Recall per Orientation (SOTA vs. GT Baby)")
    for i, c in enumerate(xs):
        ax.text(i, min(0.98, recalls[c] + 0.02), f"{recalls[c]:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "recall_per_orientation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Compute IoU statistics and generate plots
    if len(all_iou) > 0:
        iou_arr = np.asarray(all_iou, dtype=np.float32)
        iou_stats = {
            "count": int(iou_arr.size),
            "mean": float(iou_arr.mean()),
            "median": float(np.median(iou_arr)),
            "p25": float(np.percentile(iou_arr, 25)),
            "p75": float(np.percentile(iou_arr, 75)),
            "std": float(iou_arr.std(ddof=0)),
        }
    else:
        iou_stats = {"count": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "std": 0.0}

    # Save IoU statistics to a JSON file
    with open(out_dir / "iou_stats.json", "w") as jf:
        json.dump(iou_stats, jf, indent=2)

    # Generate IoU histogram
    if len(all_iou) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(all_iou, bins=20, range=(0, 1), edgecolor="black")
        ax.set_xlabel("IoU (TP)")
        ax.set_ylabel("Count")
        ax.set_title("IoU Distribution (TP)")
        fig.tight_layout()
        fig.savefig(out_dir / "iou_hist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Generate IoU boxplots per class
    if len(iou_data_for_boxplot) > 0:
        fig_bp = plot_boxplots(
            data=iou_data_for_boxplot, x_field="class", y_field="IoU",
            title="IoU per Orientation (TP)", labels_map=LABELS_MAP, y_lim=(0.0, 1.0), cmap_name="tab10",
        )
        fig_bp.savefig(out_dir / "iou_boxplot_per_class.png", dpi=150, bbox_inches="tight")
        plt.close(fig_bp)

    # Generate angle error boxplots per class (only for PCN predictions)
    if not aabb_mode and len(angle_data_for_boxplot) > 0:
        ang_fig = plot_boxplots(
            data=angle_data_for_boxplot,
            x_field="class",
            y_field="error°",
            title="Angle-Error per Orientation (TP) [PCN]",
            labels_map=LABELS_MAP,
            y_lim=(0, 180),
        )
        ang_fig.savefig(out_dir / "angle_boxplot_per_class.png", dpi=150, bbox_inches="tight")
        plt.close(ang_fig)

    # Write summary metrics to a CSV file
    with open(out_dir / "sota_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["AP_face", f"{ap_face:.4f}"])
        w.writerow(["mAP_face", f"{mAP:.4f}"])
        w.writerow([])
        w.writerow(["IoU_count_TP", iou_stats["count"]])
        w.writerow(["IoU_mean", f"{iou_stats['mean']:.4f}"])
        w.writerow(["IoU_median", f"{iou_stats['median']:.4f}"])
        w.writerow(["IoU_p25", f"{iou_stats['p25']:.4f}"])
        w.writerow(["IoU_p75", f"{iou_stats['p75']:.4f}"])
        w.writerow(["IoU_std", f"{iou_stats['std']:.4f}"])
        if not aabb_mode and any(angle_errs_per_cls[c] for c in LABELS_MAP):
            # Add angle error statistics if available
            all_angles = [v for c in angle_errs_per_cls for v in angle_errs_per_cls[c]]
            if len(all_angles) > 0:
                w.writerow([])
                w.writerow(["Angle_count_TP", len(all_angles)])
                w.writerow(["Angle_mean_deg", f"{np.mean(all_angles):.2f}"])
                w.writerow(["Angle_std_deg",  f"{np.std(all_angles):.2f}"])
        w.writerow([])
        w.writerow(["class", "GT", "TP", "FN", "Recall"])
        for c in LABELS_MAP:
            w.writerow([LABELS_MAP[c], gt_per_cls[c], tp_per_cls[c], fn_per_cls[c], f"{recalls[c]:.4f}"])
        w.writerow([])
        w.writerow(["FP_global", fp_global])

    # Print summary metrics to the console
    print("\n[RESULTS]")
    print(f"  AP (face/no-face): {ap_face:.4f}")
    for c in LABELS_MAP:
        print(f"  {LABELS_MAP[c]:<15s}  GT:{gt_per_cls[c]:4d}  TP:{tp_per_cls[c]:4d}  FN:{fn_per_cls[c]:4d}  Recall:{recalls[c]:.3f}")
    print(f"  FP global (all images): {fp_global}")
    if len(all_iou) > 0:
        print(f"  IoU(TP): mean={iou_stats['mean']:.3f}, median={iou_stats['median']:.3f}, p25={iou_stats['p25']:.3f}, p75={iou_stats['p75']:.3f}, std={iou_stats['std']:.3f}")
    if not aabb_mode and any(angle_errs_per_cls[c] for c in LABELS_MAP):
        mu = np.mean([v for c in angle_errs_per_cls for v in angle_errs_per_cls[c]])
        sd = np.std([v for c in angle_errs_per_cls for v in angle_errs_per_cls[c]])
        print(f"  Angle-Error(TP) [deg]: mean={mu:.2f}, std={sd:.2f}")

    # Return detailed metrics as a dictionary
    return {
        "AP_face": ap_face,
        "mAP_face": mAP,
        "per_true_face": per_true_face,
        "per_score_face": per_score_face,
        "gt_per_cls": gt_per_cls,
        "tp_per_cls": tp_per_cls,
        "fn_per_cls": fn_per_cls,
        "recalls": recalls,
        "fp_global": fp_global,
        "iou_stats": iou_stats,
        "angle_errs_per_cls": angle_errs_per_cls if not aabb_mode else None,
    }


def evaluate_yolo_oriented(
    data_root: Path,
    split: str,
    pred_dir: Path,
    out_dir: Path,
    iou_th: float = 0.5,
    min_score: float = 0.0,
) -> Dict[str, Any]:
    """
    Compara el modelo YOLO-based (con orientación) contra GT bebé (orientaciones 0..4).
    Genera: PR/AP/mAP, CM, IoU/Δθ boxplots, F1 vs threshold, y CSV con TP/FP/FN por clase.
    """
    images_dir = data_root / split / "images"
    labels_dir = data_root / split / "labels"
    out_dir = Path(out_dir)
    figs_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Acumuladores al estilo run_inference
    per_true = {c: [] for c in LABELS_MAP}
    per_score = {c: [] for c in LABELS_MAP}
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in LABELS_MAP}
    iou_errs = {c: [] for c in LABELS_MAP}
    angle_errs = {c: [] for c in LABELS_MAP}
    y_true: List[int] = []
    y_pred: List[int] = []
    all_gts: List[int] = []
    all_preds: List[int] = []
    all_scores: List[float] = []

    def ensure_present_for_all_classes():
        # Igual que en tu pipeline: si una clase quedó sin entradas, mete (0, 0.0) para no romper plots.
        for cls in LABELS_MAP:
            if not per_true[cls]:
                per_true[cls].append(0)
                per_score[cls].append(0.0)

    def img_size(p: Path) -> Tuple[int, int]:
        with Image.open(p) as im:
            return im.size  # (W,H)

    # Recorre imágenes
    jpgs = sorted(list(images_dir.glob("*.jpg"))) + sorted(
        list(images_dir.glob("*.png"))
    )
    for img_p in jpgs:
        stem = img_p.stem
        gt_p = labels_dir / f"{stem}.txt"
        pr_p = pred_dir / f"{stem}.txt"

        W, H = img_size(img_p)

        # ---- GT bebés ----
        gt_xywhr, gt_cls = read_gt_baby_xywhr(gt_p, (W, H))
        G = int(gt_xywhr.shape[0])

        # ---- Predicciones YOLO-based ----
        pr_xywhr, pr_cls, pr_scores = read_yolo_oriented_preds_xywhr(
            pr_p, min_score=min_score
        )
        P = int(pr_xywhr.shape[0])

        # Caso sin GT bebés: toda pred es FP (por su clase)
        # ---- Caso sin GT bebés ----
        if G == 0:
            n_adults = count_adults_in_gt(gt_p)
            if P == 0:
                # TN: adultos (uno por adulto) o 1 si es puro fondo (no hay .txt)
                tn_count = n_adults if n_adults > 0 else 1
                for _ in range(tn_count):
                    y_true.append(-1)
                    y_pred.append(-1)
                    all_gts.append(-1)
                    all_preds.append(-1)
                    all_scores.append(0.0)
            else:
                # Predicciones en imagen sin bebés ⇒ todas FP
                for j in range(P):
                    c_det = int(pr_cls[j])
                    s_det = float(pr_scores[j])
                    if c_det in stats:
                        stats[c_det]["fp"] += 1
                        per_true[c_det].append(0)
                        per_score[c_det].append(s_det)
                    y_true.append(-1)
                    y_pred.append(c_det)
                    all_gts.append(-1)
                    all_preds.append(c_det)
                    all_scores.append(s_det)
            continue
        # ---- Matching por IoU ----
        matches, unmatched_gt, unmatched_pr = greedy_match(
            gt_xywhr, pr_xywhr, pr_scores, iou_th=iou_th
        )

        matched_gt_idx = {g for (g, _, _) in matches}
        matched_pr_idx = {p for (_, p, _) in matches}

        # ---- Procesar matches ----
        for gi, pj, iou_val in matches:
            true_cls = int(gt_cls[gi])
            pred_cls = int(pr_cls[pj])
            score_det = float(pr_scores[pj])

            if pred_cls == true_cls and true_cls in stats:
                # TP clase correcta
                stats[true_cls]["tp"] += 1
                per_true[true_cls].append(1)
                per_score[true_cls].append(score_det)
                y_true.append(true_cls)
                y_pred.append(true_cls)

                # IoU y error angular
                iou_errs[true_cls].append(float(iou_val))
                dtheta = wrap_to_pi(pr_xywhr[pj, 4] - gt_xywhr[gi, 4])
                angle_errs[true_cls].append(float(torch.abs(dtheta) * 180.0 / np.pi))

                all_gts.append(true_cls)
                all_preds.append(true_cls)
                all_scores.append(score_det)

            else:
                # Confusión de clase: FP para pred_cls, FN para true_cls
                if pred_cls in stats:
                    stats[pred_cls]["fp"] += 1
                    per_true[pred_cls].append(0)
                    per_score[pred_cls].append(score_det)

                if true_cls in stats:
                    stats[true_cls]["fn"] += 1

                y_true.append(true_cls)
                y_pred.append(pred_cls)
                all_gts.append(true_cls)
                all_preds.append(pred_cls)
                all_scores.append(score_det)

        # ---- Predicciones no emparejadas → FP ----
        for pj in unmatched_pr:
            c_det = int(pr_cls[pj])
            s_det = float(pr_scores[pj])
            if c_det in stats:
                stats[c_det]["fp"] += 1
                per_true[c_det].append(0)
                per_score[c_det].append(s_det)

            y_true.append(-1)
            y_pred.append(c_det)
            all_gts.append(-1)
            all_preds.append(c_det)
            all_scores.append(s_det)

        # ---- GT no emparejados → FN ----
        for gi in unmatched_gt:
            c_gt = int(gt_cls[gi])
            if c_gt in stats:
                stats[c_gt]["fn"] += 1
                # Para PR por clase: marcar que existe el GT con score 0
                per_true[c_gt].append(1)
                per_score[c_gt].append(0.0)

            y_true.append(c_gt)
            y_pred.append(-1)
            all_gts.append(c_gt)
            all_preds.append(-1)
            all_scores.append(0.0)

    # ==========
    # Compute Metrics
    # ==========

    # Ensure all classes are represented in the metrics, even if no predictions exist for them
    ensure_present_for_all_classes()

    # Compute mean Average Precision (mAP) and AP per class
    mAP, APs = compute_map_and_pr(per_true, per_score)

    # Plot Precision-Recall (PR) curves for each class and save the figure
    pr_fig = plot_precision_recall(per_true, per_score, LABELS_MAP, mAP=mAP)
    pr_fig.savefig(figs_dir / "precision_recall.png", dpi=150, bbox_inches="tight")
    plt.close(pr_fig)

    # Plot confusion matrices (raw and normalized) and save the figures
    cm_figs = plot_confusion_matrix(y_true=y_true, y_pred=y_pred, labels_map=LABELS_MAP)
    cm_figs["raw"].savefig(figs_dir / "class_cm_raw.png", dpi=150, bbox_inches="tight")
    plt.close(cm_figs["raw"])
    cm_figs["normalized"].savefig(
        figs_dir / "class_cm_normalized.png", dpi=150, bbox_inches="tight"
    )
    plt.close(cm_figs["normalized"])

    # Plot IoU distribution as boxplots for each class and save the figure
    iou_data = [
        {"class": LABELS_MAP[c], "iou": v} for c, vals in iou_errs.items() for v in vals
    ]
    if len(iou_data) > 0:
        iou_fig = plot_boxplots(
            iou_data,
            x_field="class",
            y_field="iou",
            title="IoU Distribution per Class (OBBabyFace)",
            labels_map=LABELS_MAP,
            y_lim=(0, 1),
        )
        iou_fig.savefig(figs_dir / "iou_boxplot.png", dpi=150, bbox_inches="tight")
        plt.close(iou_fig)

    # Plot angular error distribution as boxplots for each class and save the figure
    ang_data = [
        {"class": LABELS_MAP[c], "error°": v}
        for c, vals in angle_errs.items()
        for v in vals
    ]
    if len(ang_data) > 0:
        ang_fig = plot_boxplots(
            ang_data,
            x_field="class",
            y_field="error°",
            title="Angle-Error Distribution per Class (OBBabyFace)",
            labels_map=LABELS_MAP,
            y_lim=(0, 180),
        )
        ang_fig.savefig(figs_dir / "angle_boxplot.png", dpi=150, bbox_inches="tight")
        plt.close(ang_fig)

    # Plot F1 score vs. threshold curve and save the figure
    f1_fig = plot_f1_vs_threshold(
        all_gts=all_gts,
        all_scores=all_scores,
        all_preds=all_preds,
        labels_map=LABELS_MAP,
    )
    f1_fig.savefig(figs_dir / "f1_threshold.png", dpi=150, bbox_inches="tight")
    plt.close(f1_fig)

    # ======================
    # Write Detailed Metrics to CSV
    # ======================

    # Compute raw and normalized confusion matrices for all classes (including background)
    labels_full = list(LABELS_MAP.keys()) + [-1]
    names_full = [LABELS_MAP.get(l, "BG") for l in labels_full]
    cm_raw = confusion_matrix(y_true, y_pred, labels=labels_full)
    cm_norm = cm_raw.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums != 0)

    # Helper function for safe division
    def safe_div(a, b):
        return (a / b) if b > 0 else 0.0

    # Compute mean and standard deviation for IoU and angular errors per class
    iou_mean = {c: (float(np.mean(v)) if len(v) else 0.0) for c, v in iou_errs.items()}
    iou_std = {c: (float(np.std(v)) if len(v) else 0.0) for c, v in iou_errs.items()}
    ang_mean = {
        c: (float(np.mean(v)) if len(v) else 0.0) for c, v in angle_errs.items()
    }
    ang_std = {c: (float(np.std(v)) if len(v) else 0.0) for c, v in angle_errs.items()}

    # Build rows for per-class metrics using the confusion matrix
    per_class_rows = []
    for idx_i, cls in enumerate(labels_full):
        name = LABELS_MAP.get(cls, "BG")
        TP = int(cm_raw[idx_i, idx_i])
        FP = int(cm_raw[:, idx_i].sum() - TP)
        FN = int(cm_raw[idx_i, :].sum() - TP)
        prec = safe_div(TP, TP + FP)
        rec = safe_div(TP, TP + FN)
        f1 = safe_div(2 * prec * rec, (prec + rec)) if (prec + rec) > 0 else 0.0
        ap_pr = float(APs.get(cls, 0.0)) if cls in APs else 0.0
        miou = iou_mean.get(cls, 0.0)
        siou = iou_std.get(cls, 0.0)
        mang = ang_mean.get(cls, 0.0)
        sang = ang_std.get(cls, 0.0)

        per_class_rows.append(
            [
                name,
                TP,
                FP,
                FN,
                f"{prec:.4f}",
                f"{rec:.4f}",
                f"{f1:.4f}",
                f"{ap_pr:.4f}",
                f"{miou:.4f}",
                f"{siou:.4f}",
                f"{mang:.4f}",
                f"{sang:.4f}",
            ]
        )

    # Write all metrics to a single CSV file
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)

        # Write global metrics
        w.writerow(["metric", "value"])
        w.writerow(["mAP", f"{mAP:.4f}"])
        w.writerow([])

        # Write per-class metrics
        w.writerow(["# --- METRICS PER CLASS ---"])
        w.writerow(
            [
                "Class",
                "TP",
                "FP",
                "FN",
                "Precision",
                "Recall",
                "F1",
                "AP_PR",
                "IoU_mean",
                "IoU_std",
                "Angle_mean_deg",
                "Angle_std_deg",
            ]
        )
        w.writerows(per_class_rows)
        w.writerow([])

        # Write raw confusion matrix
        w.writerow(["# --- CONFUSION MATRIX RAW ---"])
        w.writerow([""] + names_full)
        for i, rname in enumerate(names_full):
            w.writerow([rname] + [int(v) for v in cm_raw[i].tolist()])
        w.writerow([])

        # Write normalized confusion matrix
        w.writerow(["# --- CONFUSION MATRIX NORMALIZED ---"])
        w.writerow([""] + names_full)
        for i, rname in enumerate(names_full):
            w.writerow([rname] + [f"{float(v):.4f}" for v in cm_norm[i].tolist()])

    print(f"[INFO] Wrote consolidated metrics to {csv_path}")

    # Print summary of results to the console
    print("\n[RESULTS - YOLO Oriented]")
    print(f"  mAP: {mAP:.4f}")
    for c in LABELS_MAP:
        ap_c = APs.get(c, 0.0)
        print(
            f"  {LABELS_MAP[c]:<15s}  AP:{ap_c:6.3f}  TP:{stats[c]['tp']:4d}  FP:{stats[c]['fp']:4d}  "
            f"FN:{stats[c]['fn']:4d}  IoUμ:{(np.mean(iou_errs[c]) if iou_errs[c] else 0):.3f}  "
            f"Δθμ°:{(np.mean(angle_errs[c]) if angle_errs[c] else 0):.1f}"
        )

    # Return detailed metrics as a dictionary
    return {
        "mAP": mAP,
        "APs": APs,
        "stats": stats,
        "iou_errs": iou_errs,
        "angle_errs": angle_errs,
        "y_true": y_true,
        "y_pred": y_pred,
        "all_gts": all_gts,
        "all_preds": all_preds,
        "all_scores": all_scores,
    }


def main():
    ap = argparse.ArgumentParser(
        "Evaluate SOTA (face/no-face) against GT baby + orientations"
    )
    ap.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory of the dataset (contains test/images and test/labels)",
    )
    ap.add_argument(
        "--split", type=str, default="test", help="Dataset split to evaluate"
    )
    ap.add_argument(
        "--sota-dir",
        type=str,
        required=True,
        help="Directory containing SOTA predictions in .txt format",
    )
    ap.add_argument(
        "--yolo_obb",
        action="store_true",
        help="Whether to evaluate YOLO-based oriented model",
    )
    ap.add_argument(
        "--aabb_mode",
        action="store_true",
        help="Whether to evaluate YOLO-based oriented model",
    )
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching")
    ap.add_argument(
        "--min-score", type=float, default=0.0, help="Filter predictions by score"
    )
    args = ap.parse_args()

    if args.yolo_obb:
        if not args.sota_dir:
            raise ValueError(
                "For --yolo_obb evaluation, --sota-dir (predictions) is required."
            )
        evaluate_yolo_oriented(
            data_root=Path(args.data_root),
            split=args.split,
            pred_dir=Path(args.sota_dir),
            out_dir=Path(args.out),
            iou_th=args.iou,
            min_score=args.min_score,
        )
    else:
        evaluate_sota(
            data_root=Path(args.data_root),
            split=args.split,
            sota_dir=Path(args.sota_dir),
            out_dir=Path(args.out),
            iou_th=args.iou,
            min_score=args.min_score,
            aabb_mode=args.aabb_mode,
        )


if __name__ == "__main__":
    main()
