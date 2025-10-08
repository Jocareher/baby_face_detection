# evaluate_sota.py
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import Counter
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
    read_retinababyface_preds_xywhr,
    read_pcn_preds_xywhr,
    count_adults_in_gt,
    compute_loc_curves_from_predictions,
    plot_precision_recall_vs_threshold,
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


# Function to get image size
def img_size(p: Path) -> Tuple[int, int]:
    with Image.open(p) as im:
        return im.size


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
    Evaluate the performance of a state-of-the-art (SOTA) face detection model on a dataset.
    This function computes various metrics to evaluate the performance of a face detection model,
    including precision, recall, F1-score, IoU statistics, and per-class metrics. It also generates
    visualizations such as precision-recall curves, IoU histograms, and boxplots for IoU and angle errors.
    Args:
        data_root (Path): Root directory of the dataset.
        split (str): Dataset split to evaluate (e.g., "train", "val", "test").
        sota_dir (Path): Directory containing the SOTA model predictions.
        out_dir (Path): Directory to save evaluation results and visualizations.
        iou_th (float, optional): IoU threshold for matching predictions to ground truth. Defaults to 0.5.
        min_score (float, optional): Minimum confidence score for predictions to be considered. Defaults to 0.0.
        aabb_mode (bool, optional): If True, evaluate using axis-aligned bounding boxes (AABB).
            Otherwise, evaluate using oriented bounding boxes. Defaults to False.
    Returns:
        Dict[str, Any]: A dictionary containing the following evaluation results:
            - "AP_face" (float): Average precision for face/no-face classification.
            - "mAP_face" (float): Mean average precision for face/no-face classification.
            - "per_true_face" (Dict[int, List[int]]): True positive indicators for face/no-face classification.
            - "per_score_face" (Dict[int, List[float]]): Confidence scores for face/no-face classification.
            - "gt_per_cls" (Dict[int, int]): Ground truth counts per class (orientation).
            - "tp_per_cls" (Dict[int, int]): True positive counts per class (orientation).
            - "fn_per_cls" (Dict[int, int]): False negative counts per class (orientation).
            - "recalls" (Dict[int, float]): Recall values per class (orientation).
            - "fp_per_cls_loc" (Dict[int, float]): Fractional false positives per class (localization-only).
            - "prec_loc_per_cls" (Dict[int, float]): Precision values per class (localization-only).
            - "f1_loc_per_cls" (Dict[int, float]): F1-scores per class (localization-only).
            - "fp_global" (int): Total false positives across all images.
            - "iou_stats" (Dict[str, float]): IoU statistics (mean, median, percentiles, etc.).
            - "angle_errs_per_cls" (Dict[int, List[float]] or None): Angle errors per class (if not in AABB mode).
            - "n_gt_total" (int): Total number of ground truth instances.
            - "best_conf_th_loc" (float): Best confidence threshold for localization-only metrics.
            - "best_precision_loc" (float): Best precision for localization-only metrics.
            - "best_recall_loc" (float): Best recall for localization-only metrics.
            - "best_f1_loc" (float): Best F1-score for localization-only metrics.
            - "precision_loc_at_min" (float): Precision at the minimum confidence score.
            - "recall_loc_at_min" (float): Recall at the minimum confidence score.
            - "f1_loc_at_min" (float): F1-score at the minimum confidence score.
    Notes:
        - The function assumes that the dataset is organized with "images" and "labels" subdirectories
          under the specified `data_root` and `split`.
        - Ground truth labels and predictions are expected to be in specific formats, and helper functions
          like `read_gt_baby_xywhr`, `read_sota_preds_xywhr_xyxy`, and `read_pcn_preds_xywhr` are used
          to parse them.
        - The function generates various plots and saves them to the specified `out_dir`.
    Raises:
        FileNotFoundError: If the required directories or files are not found.
        ValueError: If there are inconsistencies in the input data or parameters.
    """

    # Create necessary directories
    images_dir = data_root / split / "images"
    labels_dir = data_root / split / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare accumulators for metrics (face/no-face)
    per_true_face: Dict[int, List[int]] = {0: []}
    per_score_face: Dict[int, List[float]] = {0: []}

    # TP/FP/FN per class (orientations)
    gt_per_cls = {c: 0 for c in LABELS_MAP}
    tp_per_cls = {c: 0 for c in LABELS_MAP}
    fn_per_cls = {c: 0 for c in LABELS_MAP}

    # FP per class (for loc-only precision)
    fp_per_cls = {c: 0.0 for c in LABELS_MAP}

    # Global FP (all images)
    fp_global = 0

    # IoU and angle errors (for TPs)
    all_iou: List[float] = []
    all_scores_matched: List[float] = []
    iou_per_cls: Dict[int, List[float]] = {c: [] for c in LABELS_MAP}
    iou_data_for_boxplot: List[Dict[str, Any]] = []
    angle_errs_per_cls: Dict[int, List[float]] = {c: [] for c in LABELS_MAP}
    angle_data_for_boxplot: List[Dict[str, Any]] = []

    # List all image files
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    jpgs: List[Path] = []

    # Accumulate all image paths
    for pat in exts:
        jpgs += list(images_dir.glob(pat))
    jpgs = sorted(jpgs)

    # Process each image
    for img_p in jpgs:
        # Corresponding GT and prediction files
        stem = img_p.stem
        # Read GT and prediction paths
        gt_p = labels_dir / f"{stem}.txt"
        pr_p = sota_dir / f"{stem}.txt"

        # Get image dimensions
        W, H = img_size(img_p)

        # GT baby faces
        gt_xywhr, gt_cls = read_gt_baby_xywhr(gt_p, (W, H))
        # Filter only baby faces
        if gt_cls.numel() > 0:
            # Exclude non-baby faces (cls == -1)
            keep_baby = gt_cls != -1
            # Keep only baby faces
            gt_xywhr_baby = gt_xywhr[keep_baby]
            # Get corresponding classes
            gt_cls_baby = gt_cls[keep_baby]
        else:
            # No GT baby faces
            gt_xywhr_baby = gt_xywhr
            gt_cls_baby = gt_cls

        # Count GT per class
        cls_counts_img = Counter(gt_cls_baby.tolist())
        for c in gt_cls_baby.tolist():
            gt_per_cls[int(c)] += 1

        # AABB mode when evaluating AABB predictions
        if aabb_mode:
            pr_xywhr, pr_scores = read_sota_preds_xywhr_xyxy(
                pred_txt_path=pr_p, img_wh=(W, H), min_score=min_score
            )
        else:
            # Read PCN predictions (oriented boxes)
            pr_xywhr, pr_scores = read_pcn_preds_xywhr(
                pred_txt_path=pr_p, img_wh=(W, H), min_score=min_score
            )

        # No baby GT case (only adults or empty)
        if gt_xywhr_baby.numel() == 0:
            # Count adults in GT for TN
            nfp = int(pr_xywhr.shape[0])
            # If no predictions, count TNs
            fp_global += nfp
            for s in pr_scores.tolist():
                # All predictions are FPs
                per_true_face[0].append(0)
                # Record their scores
                per_score_face[0].append(float(s))
            continue

        # Match predictions to GT baby faces
        matches, unmatched_gt, unmatched_pr = greedy_match(
            gt_xywhr_baby, pr_xywhr, pr_scores, iou_th=iou_th
        )

        # Matched predictions indices
        matched_pr_idx = set([m for (_, m, _) in matches])
        for j, s in enumerate(pr_scores.tolist()):
            # TP if matched, else FP
            per_true_face[0].append(1 if j in matched_pr_idx else 0)
            # Record their scores
            per_score_face[0].append(float(s))

        # TPs per class and IoU/angle errors
        for gi, pj, iou in matches:
            # Class of the matched GT
            c = int(gt_cls_baby[gi].item())
            # Label name
            class_name = LABELS_MAP[c]
            # Record IoU and score
            iou_val = float(iou)
            score_val = float(pr_scores[pj].item())
            # Accumulate IoU and scores
            all_iou.append(iou_val)
            all_scores_matched.append(score_val)
            iou_per_cls[c].append(iou_val)
            # Data for boxplot
            iou_data_for_boxplot.append({"class": class_name, "IoU": iou_val})

            # Angle error if not in AABB mode
            if not aabb_mode:
                # Calculate angle difference
                dtheta = wrap_to_pi(pr_xywhr[pj, 4] - gt_xywhr_baby[gi, 4])
                # Convert to degrees
                err_deg = float(torch.abs(dtheta) * 180.0 / math.pi)
                # Accumulate angle errors
                angle_errs_per_cls[c].append(err_deg)
                # Data for boxplot
                angle_data_for_boxplot.append({"class": class_name, "error°": err_deg})
            tp_per_cls[c] += 1

        # FNs per class
        for gi in unmatched_gt:
            # Class of the unmatched GT
            c = int(gt_cls_baby[gi].item())
            fn_per_cls[c] += 1

        # Global FP counting (unmatched predictions)
        n_unmatched = len(unmatched_pr)
        if n_unmatched > 0:
            # All unmatched are FPs
            fp_global += n_unmatched
            # Distribute FPs proportionally to GT class distribution
            total_babies_img = sum(cls_counts_img.values())
            # If there are baby faces in GT
            if total_babies_img > 0:
                # Distribute FPs proportionally
                for c, cnt in cls_counts_img.items():
                    fp_per_cls[int(c)] += n_unmatched * (cnt / total_babies_img)

    #  Metrics computation
    mAP, APs = compute_map_and_pr(per_true_face, per_score_face)
    ap_face = APs[0]

    # Precision-recall curve for face/no-face
    pr_fig = plot_precision_recall(
        per_true_face, per_score_face, labels_map={0: "Face"}, mAP=mAP
    )
    pr_fig.savefig(out_dir / "precision_recall_face.png", dpi=150, bbox_inches="tight")
    plt.close(pr_fig)

    # Recall per class (orientation)
    recalls = {
        c: (tp_per_cls[c] / gt_per_cls[c]) if gt_per_cls[c] > 0 else 0.0
        for c in LABELS_MAP
    }

    # Bar plot for recall per orientation
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = list(LABELS_MAP.keys())
    ax.bar([LABELS_MAP[x] for x in xs], [recalls[x] for x in xs])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title("Recall per Orientation (SOTA vs. GT Baby)")
    for i, c in enumerate(xs):
        ax.text(
            i,
            min(0.98, recalls[c] + 0.02),
            f"{recalls[c]:.2f}",
            ha="center",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "recall_per_orientation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # IoU stats
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
        iou_stats = {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "std": 0.0,
        }

    # Localization-only metrics (agnostic to class)
    n_gt_total = sum(gt_per_cls.values())
    loc_curves = compute_loc_curves_from_predictions(
        y_is_tp=per_true_face[0],
        y_scores=per_score_face[0],
        n_gt=n_gt_total,
        n_steps=200,
    )
    # Best threshold and corresponding metrics
    best_th, best_P, best_R, best_F1 = (
        loc_curves["best_th"],
        loc_curves["best_P"],
        loc_curves["best_R"],
        loc_curves["best_F1"],
    )

    # Plot F1 vs threshold
    plot_precision_recall_vs_threshold(
        th=loc_curves["thresholds"],
        prec=loc_curves["precision"],
        rec=loc_curves["recall"],
        best_th=best_th,
        out_path=(out_dir / "precision_recall_vs_threshold_loc.png"),
    )

    # Metrics at min_score threshold
    scores_np = np.asarray(per_score_face[0], dtype=np.float32)
    is_tp_np = np.asarray(per_true_face[0], dtype=np.int32)
    keep_min = scores_np >= float(min_score)
    tp_min = int((is_tp_np[keep_min] == 1).sum())
    fp_min = int((is_tp_np[keep_min] == 0).sum())
    P_min = (tp_min / (tp_min + fp_min)) if (tp_min + fp_min) > 0 else 0.0
    R_min = (tp_min / n_gt_total) if n_gt_total > 0 else 0.0
    F1_min = (2 * P_min * R_min / (P_min + R_min)) if (P_min + R_min) > 0 else 0.0

    # Precision and F1 for localization-only per class (orientation)
    prec_loc_per_cls = {}
    f1_loc_per_cls = {}
    for c in LABELS_MAP:
        tp = tp_per_cls[c]
        fp = fp_per_cls[c]  # fractional FP
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = recalls[c]
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        prec_loc_per_cls[c] = float(prec)
        f1_loc_per_cls[c] = float(f1)

    # Save IoU stats to JSON
    with open(out_dir / "iou_stats.json", "w") as jf:
        json.dump(iou_stats, jf, indent=2)

    # IoU histogram and boxplots
    if len(all_iou) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(all_iou, bins=20, range=(0, 1), edgecolor="black")
        ax.set_xlabel("IoU (TP)")
        ax.set_ylabel("Count")
        ax.set_title("IoU Distribution (TP)")
        fig.tight_layout()
        fig.savefig(out_dir / "iou_hist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Boxplots for IoU and angle errors per class
    if len(iou_data_for_boxplot) > 0:
        fig_bp = plot_boxplots(
            data=iou_data_for_boxplot,
            x_field="class",
            y_field="IoU",
            title="IoU per Orientation (TP)",
            labels_map=LABELS_MAP,
            y_lim=(0.0, 1.0),
            cmap_name="tab10",
        )
        fig_bp.savefig(
            out_dir / "iou_boxplot_per_class.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig_bp)

    # Angle error boxplots if not in AABB mode
    if not aabb_mode and len(angle_data_for_boxplot) > 0:
        ang_fig = plot_boxplots(
            data=angle_data_for_boxplot,
            x_field="class",
            y_field="error°",
            title="Angle-Error per Orientation (TP)",
            labels_map=LABELS_MAP,
            y_lim=(0, 180),
        )
        ang_fig.savefig(
            out_dir / "angle_boxplot_per_class.png", dpi=150, bbox_inches="tight"
        )
        plt.close(ang_fig)

    # CSV output
    with open(out_dir / "sota_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["AP_face", f"{ap_face:.4f}"])
        w.writerow(["mAP_face", f"{mAP:.4f}"])

        # Localization-only best threshold
        w.writerow([])
        w.writerow(["# localization-only over thresholds"])
        w.writerow(["GT_total", n_gt_total])
        w.writerow(["best_conf_threshold", f"{best_th:.4f}"])
        w.writerow(["best_precision_loc", f"{best_P:.4f}"])
        w.writerow(["best_recall_loc", f"{best_R:.4f}"])
        w.writerow(["best_f1_loc", f"{best_F1:.4f}"])

        # Localization-only at min_score
        w.writerow([])
        w.writerow(["# localization-only at min_score"])
        w.writerow(["min_score", f"{float(min_score):.4f}"])
        w.writerow(["precision_loc_at_min", f"{P_min:.4f}"])
        w.writerow(["recall_loc_at_min", f"{R_min:.4f}"])
        w.writerow(["f1_loc_at_min", f"{F1_min:.4f}"])

        # IoU stats
        w.writerow([])
        w.writerow(["IoU_count_TP", iou_stats["count"]])
        w.writerow(["IoU_mean", f"{iou_stats['mean']:.4f}"])
        w.writerow(["IoU_median", f"{iou_stats['median']:.4f}"])
        w.writerow(["IoU_p25", f"{iou_stats['p25']:.4f}"])
        w.writerow(["IoU_p75", f"{iou_stats['p75']:.4f}"])
        w.writerow(["IoU_std", f"{iou_stats['std']:.4f}"])

        # Angle error stats if available
        if not aabb_mode and any(angle_errs_per_cls[c] for c in LABELS_MAP):
            all_angles = [v for c in LABELS_MAP for v in angle_errs_per_cls[c]]
            if len(all_angles) > 0:
                w.writerow([])
                w.writerow(["Angle_count_TP", len(all_angles)])
                w.writerow(["Angle_mean_deg", f"{np.mean(all_angles):.2f}"])
                w.writerow(["Angle_std_deg", f"{np.std(all_angles):.2f}"])

        # Per-class metrics (orientation)
        w.writerow([])
        w.writerow(
            ["class", "GT", "TP", "FN", "Recall", "FP_loc", "Precision_loc", "F1_loc"]
        )
        for c in LABELS_MAP:
            w.writerow(
                [
                    LABELS_MAP[c],
                    gt_per_cls[c],
                    tp_per_cls[c],
                    fn_per_cls[c],
                    f"{recalls[c]:.4f}",
                    f"{fp_per_cls[c]:.2f}",
                    f"{prec_loc_per_cls[c]:.4f}",
                    f"{f1_loc_per_cls[c]:.4f}",
                ]
            )

        w.writerow([])
        w.writerow(["FP_global", fp_global])

    # Print summary to console
    print("\n[RESULTS]")
    print(f"  AP (face/no-face): {ap_face:.4f}")
    print(
        f"  [loc-only] best threshold = {best_th:.3f} | "
        f"P={best_P:.3f}, R={best_R:.3f}, F1={best_F1:.3f} (GT={n_gt_total})"
    )
    print(
        f"  [loc-only @min={min_score:.3f}] "
        f"P={P_min:.3f}, R={R_min:.3f}, F1={F1_min:.3f} "
        f"(TP={tp_min}, FP={fp_min}, GT={n_gt_total})"
    )
    for c in LABELS_MAP:
        print(
            f"  {LABELS_MAP[c]:<15s}  GT:{gt_per_cls[c]:4d}  TP:{tp_per_cls[c]:4d}  "
            f"FN:{fn_per_cls[c]:4d}  Recall:{recalls[c]:.3f}  |  "
            f"Precision:{prec_loc_per_cls[c]:.3f}  F1_loc:{f1_loc_per_cls[c]:.3f}"
        )
    print(f"  FP global (all images): {fp_global}")
    if len(all_iou) > 0:
        print(
            f"  IoU(TP): mean={iou_stats['mean']:.3f}, median={iou_stats['median']:.3f}, "
            f"p25={iou_stats['p25']:.3f}, p75={iou_stats['p75']:.3f}, std={iou_stats['std']:.3f}"
        )
    if not aabb_mode and any(angle_errs_per_cls[c] for c in LABELS_MAP):
        mu = np.mean([v for c in LABELS_MAP for v in angle_errs_per_cls[c]])
        sd = np.std([v for c in LABELS_MAP for v in angle_errs_per_cls[c]])
        print(f"  Angle-Error(TP) [deg]: mean={mu:.2f}, std={sd:.2f}")

    return {
        "AP_face": ap_face,
        "mAP_face": mAP,
        "per_true_face": per_true_face,
        "per_score_face": per_score_face,
        "gt_per_cls": gt_per_cls,
        "tp_per_cls": tp_per_cls,
        "fn_per_cls": fn_per_cls,
        "recalls": recalls,
        "fp_per_cls_loc": fp_per_cls,
        "prec_loc_per_cls": prec_loc_per_cls,
        "f1_loc_per_cls": f1_loc_per_cls,
        "fp_global": fp_global,
        "iou_stats": iou_stats,
        "angle_errs_per_cls": angle_errs_per_cls if not aabb_mode else None,
        "n_gt_total": n_gt_total,
        "best_conf_th_loc": best_th,
        "best_precision_loc": best_P,
        "best_recall_loc": best_R,
        "best_f1_loc": best_F1,
        "precision_loc_at_min": P_min,
        "recall_loc_at_min": R_min,
        "f1_loc_at_min": F1_min,
    }


def evaluate_obb(
    data_root: Path,
    split: str,
    pred_dir: Path,
    out_dir: Path,
    iou_th: float = 0.5,
    min_score: float = 0.0,
    model_version: str = "yolo",
) -> Dict[str, Any]:
    """
    Evaluates object bounding box (OBB) predictions against ground truth (GT) data for
    baby face detection, including strict per-class metrics and localization-only metrics.
    This function compares YOLO-oriented or RetinaBabyFace predictions with ground truth
    annotations for baby faces, considering multiple orientations. It computes various
    metrics such as precision, recall, F1-score, average precision (AP), mean average
    precision (mAP), IoU, and angle errors. Additionally, it generates visualizations
    (precision-recall curves, confusion matrices, boxplots) and writes consolidated metrics
    to a CSV file.
    Args:
        data_root (Path): Root directory containing the dataset.
        split (str): Dataset split to evaluate (e.g., "train", "val", "test").
        pred_dir (Path): Directory containing prediction files.
        out_dir (Path): Directory to save evaluation results and visualizations.
        iou_th (float, optional): IoU threshold for matching predictions with ground truth.
            Defaults to 0.5.
        min_score (float, optional): Minimum confidence score for considering predictions.
            Defaults to 0.0.
        model_version (str, optional): Model version to evaluate ("yolo" or "retinababyface").
            Defaults to "yolo".
    Returns:
        Dict[str, Any]: A dictionary containing evaluation results, including:
            - "mAP": Mean average precision.
            - "APs": Average precision per class.
            - "stats": Per-class statistics (TP, FP, FN).
            - "iou_errs": IoU errors per class.
            - "angle_errs": Angle errors per class.
            - "y_true": List of ground truth labels.
            - "y_pred": List of predicted labels.
            - "all_gts": All ground truth labels (including unmatched).
            - "all_preds": All predicted labels (including unmatched).
            - "all_scores": Confidence scores for all predictions.
            - "loc_tp_global": Global localization-only true positives.
            - "loc_fn_global": Global localization-only false negatives.
            - "recall_face_localization": Recall for face localization (global).
            - "loc_tp_per_cls": Localization-only true positives per class.
            - "loc_fn_per_cls": Localization-only false negatives per class.
            - "precision_face_localization": Precision for face localization (global).
            - "loc_fp_global": Global localization-only false positives.
            - "loc_tp_pred_cls": Localization-only true positives per predicted class.
            - "loc_fp_pred_cls": Localization-only false positives per predicted class.
            - "precision_loc_pred_per_cls": Precision for localization per predicted class.
    Notes:
        - The function assumes the presence of specific directory structures and file formats
          for images, ground truth, and predictions.
        - It generates visualizations and saves them in the specified output directory.
        - Metrics are written to a CSV file for further analysis.
    """

    # Create necessary directories
    images_dir = data_root / split / "images"
    labels_dir = data_root / split / "labels"
    # Output directories
    out_dir = Path(out_dir)
    figs_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Accumulators for metrics (strict per class)
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

    # Localization-only metrics (agnostic to class)
    loc_tp_global = 0
    loc_fn_global = 0
    loc_tp_per_cls = {c: 0 for c in LABELS_MAP}
    loc_fn_per_cls = {c: 0 for c in LABELS_MAP}

    # Localization-only FP (ignore class of prediction)
    loc_fp_global = 0
    loc_tp_pred_cls = {c: 0 for c in LABELS_MAP}
    loc_fp_pred_cls = {c: 0 for c in LABELS_MAP}

    # Ensure at least one entry for each class
    def ensure_present_for_all_classes():
        for cls in LABELS_MAP:
            if not per_true[cls]:
                per_true[cls].append(0)
                per_score[cls].append(0.0)

    # List all image files
    jpgs = sorted(list(images_dir.glob("*.jpg"))) + sorted(
        list(images_dir.glob("*.png"))
    )
    # Process each image
    for img_p in jpgs:
        # Corresponding GT and prediction files
        stem = img_p.stem
        # Read GT and prediction paths
        gt_p = labels_dir / f"{stem}.txt"
        pr_p = pred_dir / f"{stem}.txt"

        # Get image dimensions
        W, H = img_size(img_p)

        # GT baby faces
        gt_xywhr, gt_cls = read_gt_baby_xywhr(gt_p, (W, H))
        G = int(gt_xywhr.shape[0])

        # If yolo, read YOLO-oriented preds; else, read RetinaBabyFace preds
        if model_version == "yolo":
            pr_xywhr, pr_cls, pr_scores = read_yolo_oriented_preds_xywhr(
                pr_p, min_score=min_score
            )
        else:
            pr_xywhr, pr_cls, pr_scores = read_retinababyface_preds_xywhr(
                pr_p, min_score=min_score
            )
        # Number of predictions
        P = int(pr_xywhr.shape[0])

        # No GT baby case (only adults or empty)
        if G == 0:
            # Count adults in GT for TN
            n_adults = count_adults_in_gt(gt_p)
            # All predictions are FPs
            if P == 0:
                # Count TNs as one per adult (at least one if none)
                tn_count = n_adults if n_adults > 0 else 1
                for _ in range(tn_count):
                    y_true.append(-1)
                    y_pred.append(-1)
                    all_gts.append(-1)
                    all_preds.append(-1)
                    all_scores.append(0.0)
            else:
                # All predictions are FPs
                for j in range(P):
                    # Count FP (strict per class)
                    c_det = int(pr_cls[j])
                    s_det = float(pr_scores[j])
                    if c_det in stats:
                        stats[c_det]["fp"] += 1
                        per_true[c_det].append(0)
                        per_score[c_det].append(s_det)
                    # Localization-only FP (ignoring predicted class)
                    if c_det in loc_fp_pred_cls:
                        loc_fp_pred_cls[c_det] += 1
                    loc_fp_global += 1

                    y_true.append(-1)
                    y_pred.append(c_det)
                    all_gts.append(-1)
                    all_preds.append(c_det)
                    all_scores.append(s_det)
            continue

        # IoU-based greedy matching
        matches, unmatched_gt, unmatched_pr = greedy_match(
            gt_xywhr, pr_xywhr, pr_scores, iou_th=iou_th
        )

        # Localization-only TP/FP/FN counting (ignore class of prediction)
        loc_tp_global += len(matches)
        loc_fn_global += len(unmatched_gt)

        # Localization-only TP/FP/FN per class (based on GT class)
        for gi, _, _ in matches:
            c = int(gt_cls[gi])
            # Count TP for the GT class
            if c in loc_tp_per_cls:
                loc_tp_per_cls[c] += 1
        # Count FN for unmatched GT classes
        for gi in unmatched_gt:
            c = int(gt_cls[gi])
            # Count FN for the GT class
            if c in loc_fn_per_cls:
                loc_fn_per_cls[c] += 1

        # Localization-only FP per predicted class
        for _, pj, _ in matches:
            c_pred = int(pr_cls[pj])
            # Count TP for the predicted class
            if c_pred in loc_tp_pred_cls:
                loc_tp_pred_cls[c_pred] += 1

        # Count FP for unmatched predictions
        for pj in unmatched_pr:
            c_det = int(pr_cls[pj])
            if c_det in loc_fp_pred_cls:
                # Count FP for the predicted class
                loc_fp_pred_cls[c_det] += 1
            loc_fp_global += 1

        # Process matched predictions
        for gi, pj, iou_val in matches:
            true_cls = int(gt_cls[gi])
            pred_cls = int(pr_cls[pj])
            score_det = float(pr_scores[pj])
            # Strict TP/FP/FN counting per class
            if pred_cls == true_cls and true_cls in stats:
                stats[true_cls]["tp"] += 1
                per_true[true_cls].append(1)
                per_score[true_cls].append(score_det)
                y_true.append(true_cls)
                y_pred.append(true_cls)

                # Accumulate IoU and angle errors
                iou_errs[true_cls].append(float(iou_val))
                dtheta = wrap_to_pi(pr_xywhr[pj, 4] - gt_xywhr[gi, 4])
                angle_errs[true_cls].append(float(torch.abs(dtheta) * 180.0 / np.pi))
                all_gts.append(true_cls)
                all_preds.append(true_cls)
                all_scores.append(score_det)
            else:
                # Mismatch in class → FN for true class, FP for predicted class
                if pred_cls in stats:
                    stats[pred_cls]["fp"] += 1
                    per_true[pred_cls].append(0)
                    per_score[pred_cls].append(score_det)

                # Count FP for predicted class in loc-only
                if true_cls in stats:
                    stats[true_cls]["fn"] += 1
                y_true.append(true_cls)
                y_pred.append(pred_cls)
                all_gts.append(true_cls)
                all_preds.append(pred_cls)
                all_scores.append(score_det)

        # No matched predictions → FP (strict)
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

        # No matched GT → FN (strict)
        for gi in unmatched_gt:
            c_gt = int(gt_cls[gi])
            if c_gt in stats:
                stats[c_gt]["fn"] += 1
                per_true[c_gt].append(1)
                per_score[c_gt].append(0.0)
            y_true.append(c_gt)
            y_pred.append(-1)
            all_gts.append(c_gt)
            all_preds.append(-1)
            all_scores.append(0.0)

    # Finalize metrics
    ensure_present_for_all_classes()
    mAP, APs = compute_map_and_pr(per_true, per_score)

    # PR curve (strict per class)
    pr_fig = plot_precision_recall(per_true, per_score, LABELS_MAP, mAP=mAP)
    pr_fig.savefig(figs_dir / "precision_recall.png", dpi=150, bbox_inches="tight")
    plt.close(pr_fig)

    # CM raw / normalized
    cm_figs = plot_confusion_matrix(y_true=y_true, y_pred=y_pred, labels_map=LABELS_MAP)
    cm_figs["raw"].savefig(figs_dir / "class_cm_raw.png", dpi=150, bbox_inches="tight")
    plt.close(cm_figs["raw"])
    cm_figs["normalized"].savefig(
        figs_dir / "class_cm_normalized.png", dpi=150, bbox_inches="tight"
    )
    plt.close(cm_figs["normalized"])

    # Boxplots
    iou_data = [
        {"class": LABELS_MAP[c], "iou": v} for c, vs in iou_errs.items() for v in vs
    ]
    if iou_data:
        fig = plot_boxplots(
            iou_data,
            "class",
            "iou",
            "IoU Distribution per Class",
            LABELS_MAP,
            y_lim=(0, 1),
        )
        fig.savefig(figs_dir / "iou_boxplot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    ang_data = [
        {"class": LABELS_MAP[c], "error°": v}
        for c, vs in angle_errs.items()
        for v in vs
    ]
    if ang_data:
        fig = plot_boxplots(
            ang_data,
            "class",
            "error°",
            "Angle-Error Distribution per Class ",
            LABELS_MAP,
            y_lim=(0, 180),
        )
        fig.savefig(figs_dir / "angle_boxplot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # F1 vs threshold
    f1_fig = plot_f1_vs_threshold(all_gts, all_scores, all_preds, LABELS_MAP)
    f1_fig.savefig(figs_dir / "f1_threshold.png", dpi=150, bbox_inches="tight")
    plt.close(f1_fig)

    # CSV output
    labels_full = list(LABELS_MAP.keys()) + [-1]
    names_full = [LABELS_MAP.get(l, "BG") for l in labels_full]
    cm_raw = confusion_matrix(y_true, y_pred, labels=labels_full)
    cm_norm = cm_raw.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums != 0)

    # Helper for safe division
    def safe_div(a, b):
        return (a / b) if b > 0 else 0.0

    # Global localization-only metrics (agnostic to class)
    # Recall
    loc_den_rec = loc_tp_global + loc_fn_global
    recall_face_localization = (loc_tp_global / loc_den_rec) if loc_den_rec > 0 else 0.0

    # Precision
    loc_den_prec = loc_tp_global + loc_fp_global
    precision_face_localization = (
        (loc_tp_global / loc_den_prec) if loc_den_prec > 0 else 0.0
    )

    # Per class localization-only metrics (agnostic to predicted class)
    recall_loc_per_cls = {
        c: (loc_tp_per_cls[c] / (loc_tp_per_cls[c] + loc_fn_per_cls[c]))
        if (loc_tp_per_cls[c] + loc_fn_per_cls[c]) > 0
        else 0.0
        for c in LABELS_MAP
    }
    # Precision per class (agnostic to GT class, based on predicted class)
    precision_loc_pred_per_cls = {
        c: (loc_tp_pred_cls[c] / (loc_tp_pred_cls[c] + loc_fp_pred_cls[c]))
        if (loc_tp_pred_cls[c] + loc_fp_pred_cls[c]) > 0
        else 0.0
        for c in LABELS_MAP
    }

    # IoU and angle error means and stds
    iou_mean = {c: (float(np.mean(v)) if len(v) else 0.0) for c, v in iou_errs.items()}
    iou_std = {c: (float(np.std(v)) if len(v) else 0.0) for c, v in iou_errs.items()}
    ang_mean = {
        c: (float(np.mean(v)) if len(v) else 0.0) for c, v in angle_errs.items()
    }
    ang_std = {c: (float(np.std(v)) if len(v) else 0.0) for c, v in angle_errs.items()}

    # Prepare per-class rows for CSV
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

        # BG class has no loc metrics
        rloc = recall_loc_per_cls.get(cls, 0.0)
        ltp = loc_tp_per_cls.get(cls, 0)
        lfn = loc_fn_per_cls.get(cls, 0)
        ploc = precision_loc_pred_per_cls.get(cls, 0.0)
        ltp_p = loc_tp_pred_cls.get(cls, 0)
        lfp_p = loc_fp_pred_cls.get(cls, 0)

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
                f"{iou_mean.get(cls,0.0):.4f}",
                f"{iou_std.get(cls,0.0):.4f}",
                f"{ang_mean.get(cls,0.0):.4f}",
                f"{ang_std.get(cls,0.0):.4f}",
                f"{rloc:.4f}",
                ltp,
                lfn,
                f"{ploc:.4f}",
                ltp_p,
                lfp_p,
            ]
        )
    # Write consolidated CSV
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        # Global metrics
        w.writerow(["metric", "value"])
        w.writerow(["mAP", f"{mAP:.4f}"])
        w.writerow(["Recall_face_localization", f"{recall_face_localization:.4f}"])
        w.writerow(
            ["Precision_face_localization", f"{precision_face_localization:.4f}"]
        )
        w.writerow(["Loc_TP_global", loc_tp_global])
        w.writerow(["Loc_FN_global", loc_fn_global])
        w.writerow(["Loc_FP_global", loc_fp_global])
        w.writerow([])

        # Per class metrics
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
                "Recall_loc",
                "LocTP_GT",
                "LocFN_GT",
                "Precision_loc_pred",
                "LocTP_pred",
                "LocFP_pred",
            ]
        )
        w.writerows(per_class_rows)
        w.writerow([])

        # CM RAW
        w.writerow(["# --- CONFUSION MATRIX RAW ---"])
        w.writerow([""] + names_full)
        for i, rname in enumerate(names_full):
            w.writerow([rname] + [int(v) for v in cm_raw[i].tolist()])
        w.writerow([])

        # CM NORMALIZED
        w.writerow(["# --- CONFUSION MATRIX NORMALIZED ---"])
        w.writerow([""] + names_full)
        for i, rname in enumerate(names_full):
            w.writerow([rname] + [f"{float(v):.4f}" for v in cm_norm[i].tolist()])

    print(f"[INFO] Wrote consolidated metrics to {csv_path}")

    # Print summary to console
    print("\n[RESULTS - YOLO Oriented]")
    print(f"  mAP: {mAP:.4f}")

    # Global (just localization)
    print(
        f"  Recall(face, localization-only): {recall_face_localization:.4f} "
        f"({loc_tp_global}/{loc_tp_global+loc_fn_global})"
    )
    print(
        f"  Precision(face, localization-only): {precision_face_localization:.4f} "
        f"({loc_tp_global}/{loc_tp_global+loc_fp_global})"
    )

    for c in LABELS_MAP:
        name = f"{LABELS_MAP[c]:<15s}"
        ap_c = APs.get(c, 0.0)
        # LocRecall per class (anched to GT)
        loc_tp_gt = loc_tp_per_cls.get(c, 0)
        loc_fn_gt = loc_fn_per_cls.get(c, 0)
        loc_rec = (
            (loc_tp_gt / (loc_tp_gt + loc_fn_gt))
            if (loc_tp_gt + loc_fn_gt) > 0
            else 0.0
        )
        # LocPrecision per class (anched to predicted)
        loc_tp_p = loc_tp_pred_cls.get(c, 0)
        loc_fp_p = loc_fp_pred_cls.get(c, 0)
        loc_prec = (
            (loc_tp_p / (loc_tp_p + loc_fp_p)) if (loc_tp_p + loc_fp_p) > 0 else 0.0
        )

        # IoU and angle means
        iou_mu = np.mean(iou_errs[c]) if iou_errs[c] else 0.0
        ang_mu = np.mean(angle_errs[c]) if angle_errs[c] else 0.0

        print(
            f"  {name} AP: {ap_c:5.3f}  TP:{stats[c]['tp']:4d}  FP:{stats[c]['fp']:4d}  FN:{stats[c]['fn']:4d}  "
            f"LocRecall:{loc_rec:0.3f} ({loc_tp_gt}/{loc_tp_gt+loc_fn_gt})  "
            f"LocPrecision:{loc_prec:0.3f} ({loc_tp_p}/{loc_tp_p+loc_fp_p})  "
            f"IoUμ:{iou_mu:.3f}  Δθμ°:{ang_mu:.1f}"
        )

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
        "loc_tp_global": loc_tp_global,
        "loc_fn_global": loc_fn_global,
        "recall_face_localization": recall_face_localization,
        "loc_tp_per_cls": loc_tp_per_cls,
        "loc_fn_per_cls": loc_fn_per_cls,
        "precision_face_localization": precision_face_localization,
        "loc_fp_global": loc_fp_global,
        "loc_tp_pred_cls": loc_tp_pred_cls,
        "loc_fp_pred_cls": loc_fp_pred_cls,
        "precision_loc_pred_per_cls": precision_loc_pred_per_cls,
    }


def main():
    ap = argparse.ArgumentParser(
        "Evaluate SOTA (face/no-face) against GT baby + orientations"
    )
    ap.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of the dataset (contains test/images and test/labels)",
    )
    ap.add_argument(
        "--split", type=str, default="test", help="Dataset split to evaluate"
    )
    ap.add_argument(
        "--sota_dir",
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
        "--model_variant",
        type=str,
        help="Model variant: 'yolo' or 'retina' (for --yolo_obb only)",
    )
    ap.add_argument(
        "--aabb_mode",
        action="store_true",
        help="Whether to evaluate YOLO-based oriented model",
    )
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching")
    ap.add_argument(
        "--min_score", type=float, default=0.0, help="Filter predictions by score"
    )
    args = ap.parse_args()

    if args.yolo_obb:
        if not args.sota_dir:
            raise ValueError(
                "For --yolo_obb evaluation, --sota-dir (predictions) is required."
            )
        evaluate_obb(
            data_root=Path(args.data_root),
            split=args.split,
            pred_dir=Path(args.sota_dir),
            out_dir=Path(args.output_dir),
            iou_th=args.iou,
            min_score=args.min_score,
            model_version=args.model_variant,
        )
    else:
        evaluate_sota(
            data_root=Path(args.data_root),
            split=args.split,
            sota_dir=Path(args.sota_dir),
            out_dir=Path(args.output_dir),
            iou_th=args.iou,
            min_score=args.min_score,
            aabb_mode=args.aabb_mode,
        )


if __name__ == "__main__":
    main()
