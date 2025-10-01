# evaluate_sota.py
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from benchmark.benchmark import (
    read_sota_preds_xywhr_xyxy,
    greedy_match,
    read_gt_baby_xywhr,
    read_yolo_oriented_preds_xywhr,
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
):
    """
    Evaluate the performance of a State-of-the-Art (SOTA) face detection model
    against ground truth (GT) data for baby faces, including orientation-specific metrics.

    This function computes:
      - Precision-Recall (PR) and Average Precision (AP) for face/no-face detection.
      - Recall per orientation class.
      - True Positives (TP), False Positives (FP), and False Negatives (FN) globally and per class.

    Outputs:
      - Precision-Recall curve plot.
      - Recall per orientation bar plot.
      - Metrics summary in a CSV file.

    Args:
        data_root (Path): Root directory of the dataset (contains `test/images` and `test/labels`).
        split (str): Dataset split to evaluate (e.g., "test").
        sota_dir (Path): Directory containing SOTA model predictions in `.txt` format.
        out_dir (Path): Output directory for saving plots and metrics.
        iou_th (float): IoU threshold for matching predictions with ground truth. Default is 0.5.
        min_score (float): Minimum confidence score to filter predictions. Default is 0.0.

    Returns:
        dict: A dictionary containing evaluation metrics such as AP, recalls, and counts.
    """
    images_dir = data_root / split / "images"
    labels_dir = data_root / split / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accumulators for global PR (face/no-face)
    per_true_face = {0: []}  # True labels for PR computation
    per_score_face = {0: []}  # Scores for PR computation

    # Counters for orientation-specific metrics
    gt_per_cls = {c: 0 for c in LABELS_MAP}  # Ground truth counts per class
    tp_per_cls = {c: 0 for c in LABELS_MAP}  # True positives per class
    fn_per_cls = {c: 0 for c in LABELS_MAP}  # False negatives per class
    fp_global = 0  # Global false positives (all unmatched predictions)

    # Helper function to get image dimensions
    def img_size(p: Path) -> Tuple[int, int]:
        with Image.open(p) as im:
            return im.size  # (Width, Height)

    # Gather all image files in the dataset split
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    jpgs = []
    for pat in exts:
        jpgs += list(images_dir.glob(pat))
    jpgs = sorted(jpgs)

    # Loop through each image
    for img_p in jpgs:
        stem = img_p.stem
        gt_p = labels_dir / f"{stem}.txt"  # Ground truth file
        pr_p = sota_dir / f"{stem}.txt"  # Prediction file

        W, H = img_size(img_p)  # Image dimensions

        # Read ground truth (GT) data
        gt_xywhr, gt_cls = read_gt_baby_xywhr(gt_p, (W, H))
        for c in gt_cls.tolist():
            gt_per_cls[int(c)] += 1  # Count GT instances per class

        # Read SOTA predictions
        pr_xywhr, pr_scores = read_sota_preds_xywhr_xyxy(
            pred_txt_path=pr_p, img_wh=(W, H), min_score=min_score
        )

        # If no ground truth exists for the image
        if gt_xywhr.numel() == 0:
            # All predictions are false positives
            fp_global += int(pr_xywhr.shape[0])
            # Add predictions to global PR computation as negatives
            for s in pr_scores.tolist():
                per_true_face[0].append(0)
                per_score_face[0].append(float(s))
            continue

        # Perform matching between GT and predictions
        matches, unmatched_gt, unmatched_pr = greedy_match(
            gt_xywhr, pr_xywhr, pr_scores, iou_th=iou_th
        )

        # Update global PR computation
        matched_pr_idx = set([m for (_, m, _) in matches])
        for j, s in enumerate(pr_scores.tolist()):
            is_tp = 1 if j in matched_pr_idx else 0
            per_true_face[0].append(is_tp)
            per_score_face[0].append(float(s))

        # Update class-specific TP and FN counts
        matched_gt_idx = set([g for (g, _, _) in matches])
        for gi in matched_gt_idx:
            c = int(gt_cls[gi].item())
            tp_per_cls[c] += 1
        for gi in unmatched_gt:
            c = int(gt_cls[gi].item())
            fn_per_cls[c] += 1

    # ======= Compute Metrics and Save Outputs =======

    # Compute global Average Precision (AP) for face/no-face
    mAP, APs = compute_map_and_pr(per_true_face, per_score_face)
    ap_face = APs[0]

    # Plot Precision-Recall curve
    pr_fig = plot_precision_recall(
        per_true_face, per_score_face, labels_map={0: "Face"}, mAP=mAP
    )
    pr_fig.savefig(out_dir / "precision_recall_face.png", dpi=150, bbox_inches="tight")
    plt.close(pr_fig)

    # Compute recall per orientation class
    recalls = {
        c: (tp_per_cls[c] / gt_per_cls[c]) if gt_per_cls[c] > 0 else 0.0
        for c in LABELS_MAP
    }

    # Plot recall per orientation as a bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = list(LABELS_MAP.keys())
    ax.bar([LABELS_MAP[x] for x in xs], [recalls[x] for x in xs])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title("Recall per Orientation (SOTA vs. GT Baby)")
    for i, c in enumerate(xs):
        ax.text(i, recalls[c] + 0.02, f"{recalls[c]:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "recall_per_orientation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save metrics to a CSV file
    with open(out_dir / "sota_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["AP_face", f"{ap_face:.4f}"])
        w.writerow(["mAP_face", f"{mAP:.4f}"])
        w.writerow([])
        w.writerow(["class", "GT", "TP", "FN", "Recall"])
        for c in LABELS_MAP:
            w.writerow(
                [
                    LABELS_MAP[c],
                    gt_per_cls[c],
                    tp_per_cls[c],
                    fn_per_cls[c],
                    f"{recalls[c]:.4f}",
                ]
            )
        w.writerow([])
        w.writerow(["FP_global", fp_global])

    # Print results to the console
    print("\n[RESULTS]")
    print(f"  AP (face/no-face): {ap_face:.4f}")
    for c in LABELS_MAP:
        print(
            f"  {LABELS_MAP[c]:<15s}  GT:{gt_per_cls[c]:4d}  TP:{tp_per_cls[c]:4d}  "
            f"FN:{fn_per_cls[c]:4d}  Recall:{recalls[c]:.3f}"
        )
    print(f"  FP global (all images): {fp_global}")

    return {
        "AP_face": ap_face,
        "per_true_face": per_true_face,
        "per_score_face": per_score_face,
        "gt_per_cls": gt_per_cls,
        "tp_per_cls": tp_per_cls,
        "fn_per_cls": fn_per_cls,
        "recalls": recalls,
        "fp_global": fp_global,
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
        if G == 0:
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
    # Métricas
    # ==========
    ensure_present_for_all_classes()
    mAP, APs = compute_map_and_pr(per_true, per_score)

    # PR por clase + PR global
    pr_fig = plot_precision_recall(per_true, per_score, LABELS_MAP, mAP=mAP)
    pr_fig.savefig(figs_dir / "precision_recall.png", dpi=150, bbox_inches="tight")
    plt.close(pr_fig)

    # CM raw/normalized (incluye BG=-1)
    cm_figs = plot_confusion_matrix(y_true=y_true, y_pred=y_pred, labels_map=LABELS_MAP)
    cm_figs["raw"].savefig(figs_dir / "class_cm_raw.png", dpi=150, bbox_inches="tight")
    plt.close(cm_figs["raw"])
    cm_figs["normalized"].savefig(
        figs_dir / "class_cm_normalized.png", dpi=150, bbox_inches="tight"
    )
    plt.close(cm_figs["normalized"])

    # IoU boxplots por clase
    iou_data = [
        {"class": LABELS_MAP[c], "iou": v} for c, vals in iou_errs.items() for v in vals
    ]
    if len(iou_data) > 0:
        iou_fig = plot_boxplots(
            iou_data,
            x_field="class",
            y_field="iou",
            title="IoU Distribution per Class (YOLO Oriented)",
            labels_map=LABELS_MAP,
            y_lim=(0, 1),
        )
        iou_fig.savefig(figs_dir / "iou_boxplot.png", dpi=150, bbox_inches="tight")
        plt.close(iou_fig)

    # Error angular boxplots por clase
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
            title="Angle-Error Distribution per Class (YOLO Oriented)",
            labels_map=LABELS_MAP,
            y_lim=(0, 180),
        )
        ang_fig.savefig(figs_dir / "angle_boxplot.png", dpi=150, bbox_inches="tight")
        plt.close(ang_fig)

    # F1 vs threshold
    f1_fig = plot_f1_vs_threshold(
        all_gts=all_gts,
        all_scores=all_scores,
        all_preds=all_preds,
        labels_map=LABELS_MAP,
    )
    f1_fig.savefig(figs_dir / "f1_threshold.png", dpi=150, bbox_inches="tight")
    plt.close(f1_fig)

    # CSV con resumen por clase
    with open(out_dir / "yolo_oriented_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["mAP", f"{mAP:.4f}"])
        w.writerow([])
        w.writerow(["class", "AP", "TP", "FP", "FN", "mean_IoU", "mean_angle_err_deg"])
        for c in LABELS_MAP:
            ap_c = APs.get(c, 0.0)
            tp = stats[c]["tp"]
            fp = stats[c]["fp"]
            fn = stats[c]["fn"]
            miou = np.mean(iou_errs[c]) if len(iou_errs[c]) > 0 else 0.0
            mang = np.mean(angle_errs[c]) if len(angle_errs[c]) > 0 else 0.0
            w.writerow(
                [LABELS_MAP[c], f"{ap_c:.4f}", tp, fp, fn, f"{miou:.4f}", f"{mang:.2f}"]
            )

    print("\n[RESULTS - YOLO Oriented]")
    print(f"  mAP: {mAP:.4f}")
    for c in LABELS_MAP:
        ap_c = APs.get(c, 0.0)
        print(
            f"  {LABELS_MAP[c]:<15s}  AP:{ap_c:6.3f}  TP:{stats[c]['tp']:4d}  FP:{stats[c]['fp']:4d}  "
            f"FN:{stats[c]['fn']:4d}  IoUμ:{(np.mean(iou_errs[c]) if iou_errs[c] else 0):.3f}  "
            f"Δθμ°:{(np.mean(angle_errs[c]) if angle_errs[c] else 0):.1f}"
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
        "--obb",
        action="store_true",
        help="Whether to evaluate YOLO-based oriented model",
    )
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching")
    ap.add_argument(
        "--min-score", type=float, default=0.0, help="Filter predictions by score"
    )
    args = ap.parse_args()

    if args.obb:
        if not args.sota_dir:
            raise ValueError(
                "For --obb evaluation, --sota-dir (predictions) is required."
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
        )


if __name__ == "__main__":
    main()
