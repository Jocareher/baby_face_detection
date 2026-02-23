import argparse, csv
from typing import Dict, Any, List
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from benchmark.benchmark import (
    read_infantface_gt_xywhr,
    read_preds_switch,
    greedy_match,
    compute_loc_curves_from_predictions,
    plot_precision_recall_vs_threshold,
)

from utils.visualize import img_size
from loss.utils import batch_probiou


def evaluate_infantface(
    data_root: Path,
    pred_dir: Path,
    out_dir: Path,
    model_type: str,  # "retina" | "yolo" | "sota" | "pcn"
    iou_th: float = 0.5,
    min_score: float = 0.0,
) -> Dict[str, Any]:
    """
    Evaluate localization performance on the InfantFace dataset.

    This function evaluates object detection models based on localization metrics only.
    It computes precision, recall, and F1-score at various confidence thresholds, as well as IoU statistics for true positives.

    Args:
        data_root (Path): Path to the dataset root directory containing "images" and "labels" subdirectories.
        pred_dir (Path): Path to the directory containing model predictions in .txt format.
        out_dir (Path): Path to the output directory where results and plots will be saved.
        model_type (str): Type of model used for predictions. Options: "retina", "yolo", "sota", "pcn".
        iou_th (float): IoU threshold for determining true positives. Default is 0.5.
        min_score (float): Minimum confidence score for predictions to be considered. Default is 0.0.

    Returns:
        Dict[str, Any]: A dictionary containing evaluation metrics, including:
            - Total ground truth, true positives, false positives, and false negatives.
            - Precision, recall, and F1-score at the best confidence threshold and at the minimum score.
            - IoU statistics for true positives.
            - Precision-recall curves and the best confidence threshold.

    Outputs:
        - A CSV file with evaluation metrics.
        - Precision-recall curve plot.
        - IoU distribution histogram.

    Notes:
        - This function assumes that ground truth and predictions are in the same format (AABB x1 y1 x2 y2).
        - No class information is used; evaluation is localization-only.
    """
    # Define paths for images, labels, and output directories
    images_dir = data_root / "images"
    labels_dir = data_root / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Initialize variables for localization curves and IoU statistics
    y_is_tp: List[int] = (
        []
    )  # True positives (1) or false positives (0) for each prediction
    y_scores: List[float] = []  # Confidence scores for predictions
    all_iou: List[float] = []  # IoU values for true positives
    total_gt = 0  # Total ground truth boxes
    total_tp = 0  # Total true positives
    total_fp = 0  # Total false positives
    total_fn = 0  # Total false negatives

    # Enumerate all image files in the dataset
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    jpgs: List[Path] = []
    for pat in exts:
        jpgs += list(images_dir.glob(pat))
    jpgs = sorted(jpgs)

    # Process each image in the dataset
    for img_p in tqdm(jpgs, desc=f"Eval InfantFace ({model_type})"):
        stem = img_p.stem
        gt_p = labels_dir / f"{stem}.txt"  # Ground truth file
        pr_p = pred_dir / f"{stem}.txt"  # Prediction file

        W, H = img_size(img_p)  # Get image dimensions

        # Read ground truth boxes
        gt_xywhr, _ = read_infantface_gt_xywhr(gt_p)  # (G,5), dummy class
        G = int(gt_xywhr.shape[0])
        total_gt += G

        # Read model predictions
        pr_xywhr, _, pr_scores = read_preds_switch(
            model_type=model_type,
            pred_txt_path=pr_p,
            img_wh=(W, H),
            min_score=min_score,
        )
        P = int(pr_xywhr.shape[0])

        if G == 0:
            # If no ground truth, all predictions are false positives
            total_fp += P
            for s in pr_scores.tolist():
                y_is_tp.append(0)
                y_scores.append(float(s))
            continue

        # Match predictions to ground truth using IoU
        matches, unmatched_gt, unmatched_pr = greedy_match(
            gt_xywhr, pr_xywhr, pr_scores, iou_th=iou_th
        )

        # Mark true positives and false positives for precision-recall curves
        matched_pr_idx = set(pj for (_, pj, _) in matches)
        for j, s in enumerate(pr_scores.tolist()):
            y_is_tp.append(1 if j in matched_pr_idx else 0)
            y_scores.append(float(s))

        # Compute IoU for true positives and add zeros for unmatched predictions
        if pr_xywhr.numel() > 0 and gt_xywhr.numel() > 0:
            iou_mat = batch_probiou(gt_xywhr, pr_xywhr)  # (G,P)
            best_iou_per_gt = iou_mat.max(dim=1).values.cpu().numpy()
        else:
            best_iou_per_gt = np.zeros((G,), dtype=np.float32)

        # Set IoU to 0 for unmatched ground truth boxes (false negatives)
        if len(unmatched_gt) > 0:
            best_iou_per_gt = best_iou_per_gt.copy()
            best_iou_per_gt[np.array(unmatched_gt, dtype=int)] = 0.0

        # Add zeros for unmatched predictions (false positives)
        zeros_for_fp = [0.0] * len(unmatched_pr)

        # Accumulate IoU values
        all_iou.extend(best_iou_per_gt.tolist())
        all_iou.extend(zeros_for_fp)

        # Update global counters
        total_tp += len(matches)
        total_fn += len(unmatched_gt)
        total_fp += len(unmatched_pr)

    # Compute precision-recall curves and best threshold
    loc_curves = compute_loc_curves_from_predictions(
        y_is_tp=y_is_tp,
        y_scores=y_scores,
        n_gt=total_gt,
        n_steps=200,
    )
    best_th = loc_curves["best_th"]
    best_P = loc_curves["best_P"]
    best_R = loc_curves["best_R"]
    best_F1 = loc_curves["best_F1"]

    # Plot precision-recall curve
    plot_precision_recall_vs_threshold(
        th=loc_curves["thresholds"],
        prec=loc_curves["precision"],
        rec=loc_curves["recall"],
        best_th=best_th,
        out_path=(figs_dir / "precision_recall_vs_threshold_loc.png"),
    )

    # Compute metrics at the minimum score threshold
    scores_np = np.asarray(y_scores, dtype=np.float32)
    is_tp_np = np.asarray(y_is_tp, dtype=np.int32)
    keep_min = scores_np >= float(min_score)
    tp_min = int((is_tp_np[keep_min] == 1).sum())
    fp_min = int((is_tp_np[keep_min] == 0).sum())
    P_min = (tp_min / (tp_min + fp_min)) if (tp_min + fp_min) > 0 else 0.0
    R_min = (tp_min / total_gt) if total_gt > 0 else 0.0
    F1_min = (2 * P_min * R_min / (P_min + R_min)) if (P_min + R_min) > 0 else 0.0

    # Compute IoU statistics for true positives
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
        # Plot IoU distribution histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(iou_arr, bins=20, range=(0, 1), edgecolor="black")
        ax.set_xlabel("IoU")
        ax.set_ylabel("Count")
        ax.set_title(f"IoU Distribution - InfantFace")
        fig.tight_layout()
        fig.savefig(figs_dir / "iou_hist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        iou_stats = {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "std": 0.0,
        }

    # Save metrics to a CSV file
    with open(out_dir / "metrics_loc_only.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["model_type", model_type])
        w.writerow(["IoU_threshold", f"{iou_th:.3f}"])
        w.writerow(["min_score", f"{float(min_score):.4f}"])
        w.writerow([])
        w.writerow(["GT_total", total_gt])
        w.writerow(["TP_total", total_tp])
        w.writerow(["FP_total", total_fp])
        w.writerow(["FN_total", total_fn])
        w.writerow([])
        w.writerow(["# best over thresholds"])
        w.writerow(["best_conf_threshold", f"{best_th:.4f}"])
        w.writerow(["best_precision", f"{best_P:.4f}"])
        w.writerow(["best_recall", f"{best_R:.4f}"])
        w.writerow(["best_f1", f"{best_F1:.4f}"])
        w.writerow([])
        w.writerow(["# at min_score"])
        w.writerow(["precision_at_min", f"{P_min:.4f}"])
        w.writerow(["recall_at_min", f"{R_min:.4f}"])
        w.writerow(["f1_at_min", f"{F1_min:.4f}"])
        w.writerow([])
        w.writerow(["IoU_count_TP", iou_stats["count"]])
        w.writerow(["IoU_mean", f"{iou_stats['mean']:.4f}"])
        w.writerow(["IoU_median", f"{iou_stats['median']:.4f}"])
        w.writerow(["IoU_p25", f"{iou_stats['p25']:.4f}"])
        w.writerow(["IoU_p75", f"{iou_stats['p75']:.4f}"])
        w.writerow(["IoU_std", f"{iou_stats['std']:.4f}"])

    # Print results to the console
    print(f"\n[RESULTS - InfantFace LocOnly | {model_type}]")
    print(f"  GT: {total_gt}  TP:{total_tp}  FP:{total_fp}  FN:{total_fn}")
    print(
        f"  [best] th={best_th:.3f}  P={best_P:.3f}  R={best_R:.3f}  F1={best_F1:.3f}"
    )
    print(f"  [min={min_score:.3f}] P={P_min:.3f}  R={R_min:.3f}  F1={F1_min:.3f}")
    if iou_stats["count"] > 0:
        print(
            f"  IoU(TP): mean={iou_stats['mean']:.3f}  median={iou_stats['median']:.3f}  "
            f"p25={iou_stats['p25']:.3f}  p75={iou_stats['p75']:.3f}  std={iou_stats['std']:.3f}"
        )

    return {
        "n_gt_total": total_gt,
        "tp_total": total_tp,
        "fp_total": total_fp,
        "fn_total": total_fn,
        "curves": loc_curves,
        "best_conf_th": best_th,
        "best_precision": best_P,
        "best_recall": best_R,
        "best_f1": best_F1,
        "precision_at_min": P_min,
        "recall_at_min": R_min,
        "f1_at_min": F1_min,
        "iou_stats": iou_stats,
    }


def main_infantface():
    ap = argparse.ArgumentParser("Evaluate localization-only on InfantFace")
    ap.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset root (split/images, split/labels)",
    )
    ap.add_argument(
        "--pred_dir", type=str, required=True, help="Directory with predictions (.txt)"
    )
    ap.add_argument("--out_dir", type=str, required=True, help="Output dir")
    ap.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["retina", "yolo", "sota", "pcn"],
        help="Which reader to use for predictions",
    )
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min_score", type=float, default=0.5)
    args = ap.parse_args()

    evaluate_infantface(
        data_root=Path(args.data_root),
        pred_dir=Path(args.pred_dir),
        out_dir=Path(args.out_dir),
        model_type=args.model_type,
        iou_th=args.iou,
        min_score=args.min_score,
    )


if __name__ == "__main__":
    main_infantface()
