# evaluate_sota.py
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import matplotlib.pyplot as plt


from benchmark.benchmark import (
    read_preds_switch,
    greedy_match,
    read_gt_baby_xywhr,
    count_adults_in_gt,
    compute_loc_curves_from_predictions,
    plot_precision_recall_vs_threshold,
    classify_image_gt,
)
from engine.inference import (
    plot_precision_recall,
    compute_map_and_pr,
    plot_boxplots,
    plot_confusion_matrix,
    plot_f1_vs_threshold,
)
from loss.utils import batch_probiou
from data_setup.augmentations import wrap_to_pi
from utils.visualize import img_size

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
    Evaluate detection performance of a SOTA model on a baby-face dataset with:
      • Face/no-face Average Precision (AP) and PR curve.
      • Per-orientation recall (based on GT classes in LABELS_MAP).
      • FP buckets: predictions on (i) baby images, (ii) adult-only images, (iii) background-only images.
      • Localization-only summary for Scenario-1 (S1): Precision considers only FPs in baby images; Recall is micro over babies.
      • IoU computed over ALL ground truths (each GT contributes its best IoU; unmatched GT → 0.0).
      • Angle error computed over ALL ground truths (matched GT → |Δθ| deg; unmatched GT → 180°).
        - Angle is reported only when evaluating oriented predictions (i.e., aabb_mode=False).

    Parameters
    ----------
    data_root : Path
        Root directory of the dataset (contains e.g. `split/images`, `split/labels`).
    split : str
        Dataset split (e.g., "train", "val", "test").
    sota_dir : Path
        Directory that contains the model's predictions (.txt per image).
    out_dir : Path
        Directory where figures and CSV summaries will be written.
    iou_th : float, default=0.5
        IoU threshold used by the greedy matcher to declare TP matches.
    min_score : float, default=0.0
        Minimum confidence score to keep predictions.
    aabb_mode : bool, default=False
        If True, evaluate AABB predictions (θ assumed 0 on preds); if False, evaluate oriented boxes.

    Returns
    -------
    Dict[str, Any]
        A dictionary with the main aggregates, including:
          - 'AP_face': float
          - 'recalls': Dict[int,float] (per-class recall)
          - 'recall_micro': float (TP_total / GT_total over baby classes)
          - 'recall_macro': float (mean of per-class recalls)
          - 'precision_s1': float (TP / (TP + FP_in_baby_imgs))
          - 'f1_s1': float (F1 using precision_s1 and recall_micro)
          - 'fp_global', 'fp_in_baby_imgs', 'fp_in_adult_imgs', 'fp_in_bg_imgs': ints
          - 'iou_stats_all_gt': Dict[str,float] (IoU stats over all GTs)
          - 'angle_stats_all_gt': Dict[str,float] (Angle stats over all GTs; only if aabb_mode=False)
          - and raw per-class counts (gt_per_cls, tp_per_cls, fn_per_cls)
    """

    # ----------------------------- I/O setup -----------------------------
    images_dir = data_root / split / "images"  # path to images
    labels_dir = data_root / split / "labels"  # path to GT labels
    out_dir.mkdir(parents=True, exist_ok=True)  # ensure output dir
    figs_dir = out_dir / "figures"  # figures subdir
    figs_dir.mkdir(parents=True, exist_ok=True)  # ensure figures dir

    # -------------------- Accumulators: binary face/no-face --------------------
    per_true_face: Dict[int, List[int]] = {
        0: []
    }  # 1 if pred matched any baby GT, else 0
    per_score_face: Dict[int, List[float]] = {0: []}  # scores for the binary PR

    # --------------------------- Per-class counters ---------------------------
    gt_per_cls = {c: 0 for c in LABELS_MAP}  # GT count per baby class
    tp_per_cls = {c: 0 for c in LABELS_MAP}  # TP per baby class
    fn_per_cls = {c: 0 for c in LABELS_MAP}  # FN per baby class

    # ------------------------------ FP buckets -------------------------------
    fp_global = 0  # total FPs across all images
    fp_in_baby_imgs = 0  # FPs on images with baby GT
    fp_in_adult_imgs = 0  # FPs on images with only adult GT (-1)
    fp_in_bg_imgs = 0  # FPs on background-only images (no GT)

    # --------- IoU/Angle over ALL GTs (global + per-class lists) -------------
    iou_all_gts: List[float] = []  # best IoU per GT
    iou_all_gts_per_cls: Dict[int, List[float]] = {
        c: [] for c in LABELS_MAP
    }  # per-class IoUs

    angle_all_gts: List[float] = []  # |Δθ| per GT (deg), or 180 for unmatched
    angle_all_gts_per_cls: Dict[int, List[float]] = {
        c: [] for c in LABELS_MAP
    }  # per-class angles

    # --------------------------- Enumerate images ----------------------------
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")  # supported extensions
    jpgs: List[Path] = []
    for pat in exts:
        jpgs += list(images_dir.glob(pat))  # gather images
    jpgs = sorted(jpgs)  # deterministic order

    # ====================== Main evaluation loop per image =====================
    for img_p in jpgs:
        stem = img_p.stem  # file stem
        gt_p = labels_dir / f"{stem}.txt"  # GT path
        pr_p = sota_dir / f"{stem}.txt"  # prediction path
        W, H = img_size(img_p)  # image size

        # Read GT boxes and classes (adults = -1)
        gt_xywhr, gt_cls = read_gt_baby_xywhr(gt_p, (W, H))  # tensors (N,5) and (N,)

        # Keep only baby GT for recall/PR
        if gt_cls.numel() > 0:
            keep_baby = gt_cls != -1  # mask non-adult
            gt_xywhr_baby = gt_xywhr[keep_baby]  # (Gb,5)
            gt_cls_baby = gt_cls[keep_baby]  # (Gb,)
        else:
            gt_xywhr_baby = gt_xywhr  # empty tensors
            gt_cls_baby = gt_cls

        # Update per-class GT counts
        for c in gt_cls_baby.tolist():
            gt_per_cls[int(c)] += 1

        # Read predictions depending on mode (AABB vs oriented)
        model_type = "sota" if aabb_mode else "pcn"
        pr_xywhr, _, pr_scores = read_preds_switch(
            model_type=model_type,
            pred_txt_path=pr_p,
            img_wh=(W, H),
            min_score=min_score,
        )

        P = int(pr_xywhr.shape[0])  # number of predictions

        # Classify this image by its GT content (for FP buckets)
        img_kind = classify_image_gt(gt_p)

        # --------- Adult-only / BG images: all predictions are FP for binary PR ---------
        if img_kind != "BABY":
            if P > 0:  # each pred is FP
                if img_kind == "ADULT_ONLY":
                    fp_in_adult_imgs += P
                else:
                    fp_in_bg_imgs += P
                fp_global += P
                for s in pr_scores.tolist():
                    per_true_face[0].append(0)  # FP for face/no-face
                    per_score_face[0].append(float(s))
            # No IoU/angle contribution from these images (no baby GT)
            continue

        # ------------------------ Baby images: do matching ------------------------
        G = int(gt_xywhr_baby.shape[0])  # number of baby GT

        # Greedy IoU matching (TP/FP/FN on baby GT)
        matches, unmatched_gt, unmatched_pr = greedy_match(
            gt_xywhr_baby, pr_xywhr, pr_scores, iou_th=iou_th
        )

        # Binary PR: mark predictions as TP if matched, else FP
        matched_pr_idx = set(
            [pj for (gi, pj, _) in matches]
        )  # pred indices that matched
        for j, s in enumerate(pr_scores.tolist()):
            per_true_face[0].append(1 if j in matched_pr_idx else 0)
            per_score_face[0].append(float(s))

        # Per-class TP/FN accounting for recall
        for gi, pj, iou_val in matches:
            c = int(gt_cls_baby[gi].item())  # GT class
            tp_per_cls[c] += 1  # TP for that class
        for gi in unmatched_gt:
            c = int(gt_cls_baby[gi].item())  # GT class
            fn_per_cls[c] += 1  # FN for that class

        # Bucket FPs in baby images (unmatched predictions)
        n_unmatched_pr = len(unmatched_pr)
        if n_unmatched_pr > 0:
            fp_in_baby_imgs += n_unmatched_pr
            fp_global += n_unmatched_pr

        # ---------------- IoU over ALL GTs (best-over-preds; unmatched→0) ----------------
        if G > 0:
            if P > 0:
                iou_mat = batch_probiou(gt_xywhr_baby, pr_xywhr)  # (G,P) IoU matrix
                best_iou_per_gt = (
                    iou_mat.max(dim=1).values.cpu().numpy()
                )  # best IoU for each GT
            else:
                best_iou_per_gt = np.zeros((G,), dtype=np.float32)  # no preds → zeros

            # Force exact zero at unmatched GT indices
            if len(unmatched_gt) > 0 and P > 0:
                best_iou_per_gt = best_iou_per_gt.copy()
                best_iou_per_gt[np.asarray(unmatched_gt, dtype=int)] = 0.0

            # Append globally and per class
            iou_all_gts.extend(best_iou_per_gt.tolist())
            for gi in range(G):
                c = int(gt_cls_baby[gi].item())
                iou_all_gts_per_cls[c].append(float(best_iou_per_gt[gi]))

        # -------- Angle over ALL GTs (matched→|Δθ| deg; unmatched→180) --------
        if not aabb_mode and G > 0:  # only when oriented
            gi_to_pj = {gi: pj for (gi, pj, _) in matches}  # map GT → pred when matched
            for gi in range(G):
                c = int(gt_cls_baby[gi].item())  # GT class
                if gi in gi_to_pj:
                    pj = gi_to_pj[gi]  # matched pred index
                    dtheta = wrap_to_pi(pr_xywhr[pj, 4] - gt_xywhr_baby[gi, 4])
                    err_deg = float(torch.abs(dtheta) * 180.0 / math.pi)
                # else:
                #     err_deg = 0.0  # unmatched GT → worst-case angle error
                angle_all_gts.append(err_deg)  # global list
                angle_all_gts_per_cls[c].append(err_deg)  # per-class list

    # ========================== Global metrics/curves ==========================
    # Face/no-face AP
    mAP, APs = compute_map_and_pr(per_true_face, per_score_face)
    ap_face = APs[0]

    # Per-class recall and micro/macro recalls
    recalls = {
        c: (tp_per_cls[c] / gt_per_cls[c]) if gt_per_cls[c] > 0 else 0.0
        for c in LABELS_MAP
    }
    TP_baby_total = int(sum(tp_per_cls.values()))
    GT_baby_total = int(sum(gt_per_cls.values()))
    recall_micro = (TP_baby_total / GT_baby_total) if GT_baby_total > 0 else 0.0
    recalls_present = [recalls[c] for c in LABELS_MAP if gt_per_cls[c] > 0]
    recall_macro = float(np.mean(recalls_present)) if len(recalls_present) > 0 else 0.0

    # Scenario-1 summary (precision uses only FP in baby images)
    precision_s1 = (
        TP_baby_total / (TP_baby_total + fp_in_baby_imgs)
        if (TP_baby_total + fp_in_baby_imgs) > 0
        else 0.0
    )
    f1_s1 = (
        (2 * precision_s1 * recall_micro / (precision_s1 + recall_micro))
        if (precision_s1 + recall_micro) > 0
        else 0.0
    )

    # PR vs threshold (visual)
    n_gt_total = GT_baby_total
    loc_curves = compute_loc_curves_from_predictions(
        y_is_tp=per_true_face[0],
        y_scores=per_score_face[0],
        n_gt=n_gt_total,
        n_steps=200,
    )
    best_th, best_P, best_R, best_F1 = (
        loc_curves["best_th"],
        loc_curves["best_P"],
        loc_curves["best_R"],
        loc_curves["best_F1"],
    )
    plot_precision_recall_vs_threshold(
        th=loc_curves["thresholds"],
        prec=loc_curves["precision"],
        rec=loc_curves["recall"],
        best_th=best_th,
        out_path=(figs_dir / "precision_recall_vs_threshold_loc.png"),
    )

    # ============================= IoU stats/plots =============================
    if len(iou_all_gts) > 0:
        iou_arr = np.asarray(iou_all_gts, dtype=np.float32)
        iou_stats = {
            "count": int(iou_arr.size),
            "mean": float(iou_arr.mean()),
            "median": float(np.median(iou_arr)),
            "p25": float(np.percentile(iou_arr, 25)),
            "p75": float(np.percentile(iou_arr, 75)),
            "std": float(iou_arr.std(ddof=0)),
        }
        # Histogram (global, per GT)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(iou_arr, bins=20, range=(0, 1), edgecolor="black")
        ax.set_xlabel("IoU")
        ax.set_ylabel("Count")
        ax.set_title("IoU over all ground truths")
        fig.tight_layout()
        fig.savefig(figs_dir / "iou_hist_all_gt.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Boxplot per class (GT-anchored IoU)
        iou_data = [
            {"class": LABELS_MAP[c], "IoU": v}
            for c, vs in iou_all_gts_per_cls.items()
            for v in vs
        ]
        if iou_data:
            fig_bp = plot_boxplots(
                data=iou_data,
                x_field="class",
                y_field="IoU",
                title="IoU per Class",
                labels_map=LABELS_MAP,
                y_lim=(0.0, 1.0),
                cmap_name="tab10",
            )
            fig_bp.savefig(
                figs_dir / "iou_boxplot_per_class_all_gt.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig_bp)
    else:
        iou_stats = {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "std": 0.0,
        }

    # =========================== Angle stats/plots ============================
    if not aabb_mode:
        if len(angle_all_gts) > 0:
            ang_arr = np.asarray(angle_all_gts, dtype=np.float32)
            angle_stats = {
                "count": int(ang_arr.size),
                "mean_deg": float(ang_arr.mean()),
                "median_deg": float(np.median(ang_arr)),
                "p25_deg": float(np.percentile(ang_arr, 25)),
                "p75_deg": float(np.percentile(ang_arr, 75)),
                "std_deg": float(ang_arr.std(ddof=0)),
            }
            # Histogram (global, per GT)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(ang_arr, bins=20, range=(0, 180), edgecolor="black")
            ax.set_xlabel("Angle error")
            ax.set_ylabel("Count")
            ax.set_title("Angle error over all ground truths")
            fig.tight_layout()
            fig.savefig(
                figs_dir / "angle_hist_all_gt.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

            # Boxplot per class (GT-anchored |Δθ|)
            angle_data = [
                {"class": LABELS_MAP[c], "error°": v}
                for c, vs in angle_all_gts_per_cls.items()
                for v in vs
            ]
            if angle_data:
                ang_fig = plot_boxplots(
                    data=angle_data,
                    x_field="class",
                    y_field="error°",
                    title="Angle error per Class",
                    labels_map=LABELS_MAP,
                    y_lim=(0, 180),
                )
                ang_fig.savefig(
                    figs_dir / "angle_boxplot_per_class_all_gt.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(ang_fig)
        else:
            angle_stats = {
                "count": 0,
                "mean_deg": 0.0,
                "median_deg": 0.0,
                "p25_deg": 0.0,
                "p75_deg": 0.0,
                "std_deg": 0.0,
            }
    else:
        angle_stats = None  # not applicable for AABB mode

    # ============================== Extra visuals =============================
    pr_fig = plot_precision_recall(
        per_true_face, per_score_face, labels_map={0: "Face"}, mAP=mAP
    )
    pr_fig.savefig(figs_dir / "precision_recall_face.png", dpi=150, bbox_inches="tight")
    plt.close(pr_fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    xs = list(LABELS_MAP.keys())
    ax.bar([LABELS_MAP[x] for x in xs], [recalls[x] for x in xs])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title("Recall per orientation")
    for i, c in enumerate(xs):
        ax.text(
            i,
            min(0.98, recalls[c] + 0.02),
            f"{recalls[c]:.2f}",
            ha="center",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(figs_dir / "recall_per_orientation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================================ CSV dump ================================
    csv_path = out_dir / "sota_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["AP_face", f"{ap_face:.4f}"])
        w.writerow([])
        w.writerow(["# Face/no-face PR vs threshold"])
        w.writerow(["GT_total_babies", GT_baby_total])
        w.writerow(["best_conf_threshold", f"{best_th:.4f}"])
        w.writerow(["best_precision", f"{best_P:.4f}"])
        w.writerow(["best_recall", f"{best_R:.4f}"])
        w.writerow(["best_f1", f"{best_F1:.4f}"])
        w.writerow([])
        w.writerow(["# Global recall"])
        w.writerow(["recall_micro", f"{recall_micro:.4f}"])
        w.writerow(["recall_macro", f"{recall_macro:.4f}"])
        w.writerow([])
        w.writerow(["# Scenario-1 (FP only from baby images)"])
        w.writerow(["precision_s1", f"{precision_s1:.4f}"])
        w.writerow(["recall_s1(micro)", f"{recall_micro:.4f}"])
        w.writerow(["f1_s1", f"{f1_s1:.4f}"])
        w.writerow([])
        w.writerow(["# FP buckets"])
        w.writerow(["fp_in_baby_imgs", fp_in_baby_imgs])
        w.writerow(["fp_in_adult_imgs", fp_in_adult_imgs])
        w.writerow(["fp_in_bg_imgs", fp_in_bg_imgs])
        w.writerow(["fp_global", fp_global])
        w.writerow([])
        w.writerow(["# IoU over ALL GTs"])
        w.writerow(["IoU_count_GT", iou_stats["count"]])
        w.writerow(["IoU_mean", f"{iou_stats['mean']:.4f}"])
        w.writerow(["IoU_median", f"{iou_stats['median']:.4f}"])
        w.writerow(["IoU_p25", f"{iou_stats['p25']:.4f}"])
        w.writerow(["IoU_p75", f"{iou_stats['p75']:.4f}"])
        w.writerow(["IoU_std", f"{iou_stats['std']:.4f}"])
        w.writerow([])
        if angle_stats is not None:
            w.writerow(["# Angle over ALL GTs (deg)"])
            w.writerow(["Angle_count_GT", angle_stats["count"]])
            w.writerow(["Angle_mean_deg", f"{angle_stats['mean_deg']:.2f}"])
            w.writerow(["Angle_median_deg", f"{angle_stats['median_deg']:.2f}"])
            w.writerow(["Angle_p25_deg", f"{angle_stats['p25_deg']:.2f}"])
            w.writerow(["Angle_p75_deg", f"{angle_stats['p75_deg']:.2f}"])
            w.writerow(["Angle_std_deg", f"{angle_stats['std_deg']:.2f}"])
            w.writerow([])
        w.writerow(["# Recall per class"])
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

    # ============================== Console log ===============================
    print("\n[RESULTS — SOTA vs Baby]")
    print(f"  AP (face/no-face): {ap_face:.4f}")
    print(
        f"  PR(best) face/no-face: th={best_th:.3f}  P={best_P:.3f}  R={best_R:.3f}  F1={best_F1:.3f}  (GT={GT_baby_total})"
    )
    print(f"  Recall (global): micro={recall_micro:.3f}  |  macro={recall_macro:.3f}")
    print(
        f"  [S1] Precision={precision_s1:.3f}  Recall={recall_micro:.3f}  F1={f1_s1:.3f}"
    )
    for c in LABELS_MAP:
        print(
            f"  {LABELS_MAP[c]:<15s}  GT:{gt_per_cls[c]:4d}  TP:{tp_per_cls[c]:4d}  FN:{fn_per_cls[c]:4d}  Recall:{recalls[c]:.3f}"
        )
    print(
        f"  FP buckets -> baby:{fp_in_baby_imgs}  adult_only:{fp_in_adult_imgs}  bg:{fp_in_bg_imgs}  |  FP_global:{fp_global}"
    )
    print(
        f"  IoU(all GT): mean={iou_stats['mean']:.3f}  median={iou_stats['median']:.3f}  p25={iou_stats['p25']:.3f}  p75={iou_stats['p75']:.3f}  std={iou_stats['std']:.3f}"
    )
    if angle_stats is not None:
        print(
            f"  Angle(all GT) [deg]: mean={angle_stats['mean_deg']:.2f}  median={angle_stats['median_deg']:.2f}  "
            f"p25={angle_stats['p25_deg']:.2f}  p75={angle_stats['p75_deg']:.2f}  std={angle_stats['std_deg']:.2f}"
        )

    # =============================== Return dict =============================
    return {
        "AP_face": ap_face,
        "per_true_face": per_true_face,
        "per_score_face": per_score_face,
        "gt_per_cls": gt_per_cls,
        "tp_per_cls": tp_per_cls,
        "fn_per_cls": fn_per_cls,
        "recalls": recalls,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "fp_global": fp_global,
        "fp_in_baby_imgs": fp_in_baby_imgs,
        "fp_in_adult_imgs": fp_in_adult_imgs,
        "fp_in_bg_imgs": fp_in_bg_imgs,
        "precision_s1": precision_s1,
        "f1_s1": f1_s1,
        "n_gt_total": GT_baby_total,
        "best_conf_th_loc": best_th,
        "best_precision_loc": best_P,
        "best_recall_loc": best_R,
        "best_f1_loc": best_F1,
        "iou_stats_all_gt": iou_stats,
        "angle_stats_all_gt": angle_stats,
        "iou_all_gts_per_cls": iou_all_gts_per_cls,
        "angle_all_gts_per_cls": angle_all_gts_per_cls
        if angle_stats is not None
        else None,
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

    # ---------- I/O ----------
    images_dir = data_root / split / "images"
    labels_dir = data_root / split / "labels"
    out_dir = Path(out_dir)
    figs_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Accumulators for strict multi-class PR/AP ----------
    per_true = {
        c: [] for c in LABELS_MAP
    }  # TP(1)/FP(0) flags per predicted score for PR per class
    per_score = {c: [] for c in LABELS_MAP}  # scores aligned with per_true
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in LABELS_MAP}

    # ---------- Quality (TP-only) ----------
    iou_errs = {
        c: [] for c in LABELS_MAP
    }  # IoU for matched (TP) pairs -> TP quality by class
    angle_errs = {c: [] for c in LABELS_MAP}  # Angle error for TPs (UNCHANGED)

    # ---------- GT-anchored IoU (FN -> 0) ----------
    iou_all_gt_per_cls: Dict[int, List[float]] = {
        c: [] for c in LABELS_MAP
    }  # per class
    iou_all_gt_global: List[float] = []  # global across all classes

    # ---------- For face/no-face (adult-SOTA comparable) ----------
    per_true_face = {0: []}  # TP/FP flags class-agnostic
    per_score_face = {0: []}  # scores class-agnostic
    fp_in_baby_imgs = 0
    fp_in_adult_imgs = 0
    fp_in_bg_imgs = 0
    fp_global_loc = 0  # class-agnostic FP for localization summary
    bg_instances_adult_total = 0
    bg_instances_pure_total = 0

    # ---------- Collections for CM and F1-threshold plots ----------
    y_true: List[int] = []
    y_pred: List[int] = []
    all_gts: List[int] = []
    all_preds: List[int] = []
    all_scores: List[float] = []

    # ---------- Loc-only counters (class-agnostic, GT-anchored) ----------
    loc_tp_global = 0
    loc_fn_global = 0
    loc_tp_per_cls = {c: 0 for c in LABELS_MAP}
    loc_fn_per_cls = {c: 0 for c in LABELS_MAP}
    loc_tp_pred_cls = {c: 0 for c in LABELS_MAP}
    loc_fp_pred_cls = {c: 0 for c in LABELS_MAP}

    # Helper: ensure PR vectors are not empty so AP is defined
    def ensure_present_for_all_classes():
        for cls in LABELS_MAP:
            if not per_true[cls]:
                per_true[cls].append(0)
                per_score[cls].append(0.0)

    # ---------- Enumerate images ----------
    jpgs = sorted(list(images_dir.glob("*.jpg"))) + sorted(
        list(images_dir.glob("*.png"))
    )

    for img_p in jpgs:
        stem = img_p.stem
        gt_p = labels_dir / f"{stem}.txt"
        pr_p = pred_dir / f"{stem}.txt"

        # Image size
        W, H = img_size(img_p)

        # Read GT (AABB/OBB in xywhr) with cls=-1 for adults
        gt_xywhr, gt_cls = read_gt_baby_xywhr(gt_p, (W, H))
        G = int(gt_xywhr.shape[0])

        # Read predictions (YOLO oriented or RetinaBabyFace)
        model_type = "yolo" if model_version == "yolo" else "retina"
        pr_xywhr, pr_cls, pr_scores = read_preds_switch(
            model_type=model_type,
            pred_txt_path=pr_p,
            img_wh=(W, H),
            min_score=min_score,
        )

        P = int(pr_xywhr.shape[0])

        # Classify image kind for FP bucketing
        img_kind = classify_image_gt(gt_p)

        # --------- No baby GTs (ADULT_ONLY or BG) ---------
        if img_kind != "BABY":
            # 1) Count background (BG) instances:
            #    - ADULT_ONLY: count one instance per annotated adult
            #    - Pure BG (no .txt or empty .txt): count 1 BG instance
            if img_kind == "ADULT_ONLY":
                n_bg_instances = count_adults_in_gt(gt_p)  # Use helper to count adults
                bg_instances_adult_total += n_bg_instances
                fp_in_adult_imgs += P  # All predictions are false positives
            else:  # Pure "BG" (no annotations)
                n_bg_instances = 1
                bg_instances_pure_total += 1
                fp_in_bg_imgs += P  # All predictions are false positives

            # 2) Add true negatives (TNs) for each BG instance to fill the BG/BG diagonal
            for _ in range(max(1, n_bg_instances)):
                y_true.append(-1)  # True label is background (-1)
                y_pred.append(-1)  # Predicted label is also background (-1)
                all_gts.append(-1)  # Ground truth is background
                all_preds.append(-1)  # Prediction is background
                all_scores.append(0.0)  # No confidence score for TNs

            # 3) Register false positives (FPs) if there are predictions in non-baby images
            if P > 0:
                fp_global_loc += P  # Increment global FP count
                for j in range(P):
                    c_det = int(pr_cls[j])  # Predicted class
                    s_det = float(pr_scores[j])  # Confidence score

                    # Face/no-face PR: all predictions are false positives
                    per_true_face[0].append(0)  # Mark as FP
                    per_score_face[0].append(s_det)  # Record score

                    # Strict multi-class: mark as FP for the predicted class
                    if c_det in stats:
                        stats[c_det]["fp"] += 1
                        per_true[c_det].append(0)  # Mark as FP
                        per_score[c_det].append(s_det)  # Record score

                    # Confusion matrix: row for BG (true=-1) against predicted class
                    y_true.append(-1)  # True label is background
                    y_pred.append(c_det)  # Predicted class
                    all_gts.append(-1)  # Ground truth is background
                    all_preds.append(c_det)  # Prediction is the detected class
                    all_scores.append(s_det)  # Confidence score

            # No IoU/angle calculations since there are no baby ground truths
            continue

        # --------- Images WITH baby GTs ---------
        # Match predictions to GTs by IoU and score
        matches, unmatched_gt, unmatched_pr = greedy_match(
            gt_xywhr, pr_xywhr, pr_scores, iou_th=iou_th
        )

        # Class-agnostic PR flags for adult-SOTA comparable summary
        matched_pr_idx = set(pj for (_, pj, _) in matches)
        for j, s in enumerate(pr_scores.tolist()):
            per_true_face[0].append(1 if j in matched_pr_idx else 0)
            per_score_face[0].append(float(s))

        # Localization-only (class-agnostic) counts
        loc_tp_global += len(matches)
        loc_fn_global += len(unmatched_gt)
        fp_global_loc += len(unmatched_pr)

        # Per-class loc-only GT-anchored (for recall by class)
        for gi, _, _ in matches:
            c = int(gt_cls[gi])
            if c in loc_tp_per_cls:
                loc_tp_per_cls[c] += 1
        for gi in unmatched_gt:
            c = int(gt_cls[gi])
            if c in loc_fn_per_cls:
                loc_fn_per_cls[c] += 1

        # Per-class loc-only predicted (for precision by predicted class)
        for _, pj, _ in matches:
            c_pred = int(pr_cls[pj])
            if c_pred in loc_tp_pred_cls:
                loc_tp_pred_cls[c_pred] += 1
        for pj in unmatched_pr:
            c_pred = int(pr_cls[pj])
            if c_pred in loc_fp_pred_cls:
                loc_fp_pred_cls[c_pred] += 1

        # --------- STRICT multi-class stats + quality metrics (TP-only) ---------
        for gi, pj, iou_val in matches:
            true_cls = int(gt_cls[gi])
            pred_cls = int(pr_cls[pj])
            score_det = float(pr_scores[pj])
            if pred_cls == true_cls and true_cls in stats:
                # TP strictly by class
                stats[true_cls]["tp"] += 1
                per_true[true_cls].append(1)
                per_score[true_cls].append(score_det)
                # TP-only IoU
                iou_errs[true_cls].append(float(iou_val))
                # Angle error (UNCHANGED)
                dtheta = wrap_to_pi(pr_xywhr[pj, 4] - gt_xywhr[gi, 4])
                angle_errs[true_cls].append(float(torch.abs(dtheta) * 180.0 / np.pi))
                # CM bookkeeping
                y_true.append(true_cls)
                y_pred.append(true_cls)
                all_gts.append(true_cls)
                all_preds.append(true_cls)
                all_scores.append(score_det)
            else:
                # Class mismatch → FP for predicted, FN for true
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

        # Unmatched predictions → strict FP of their predicted class
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

        # Unmatched GT → strict FN of its true class
        for gi in unmatched_gt:
            c_gt = int(gt_cls[gi])
            if c_gt in stats:
                stats[c_gt]["fn"] += 1
                # keep your previous convention to stabilize PR
                per_true[c_gt].append(1)
                per_score[c_gt].append(0.0)
            y_true.append(c_gt)
            y_pred.append(-1)
            all_gts.append(c_gt)
            all_preds.append(-1)
            all_scores.append(0.0)

        # --------- IoU over ALL GT (GT-anchored; FN -> 0) ---------
        if G > 0:
            if P > 0:
                iou_mat = batch_probiou(gt_xywhr, pr_xywhr)  # (G,P)
                best_per_gt = iou_mat.max(dim=1).values.cpu().numpy()
            else:
                best_per_gt = np.zeros((G,), dtype=np.float32)
            if len(unmatched_gt) > 0 and P > 0:
                best_per_gt = best_per_gt.copy()
                best_per_gt[np.asarray(unmatched_gt, dtype=int)] = 0.0
            for gi in range(G):
                c_gt = int(gt_cls[gi])
                if c_gt in iou_all_gt_per_cls:
                    iou_all_gt_per_cls[c_gt].append(float(best_per_gt[gi]))
                iou_all_gt_global.append(float(best_per_gt[gi]))

        # --------- Bucket FP in BABY images for S1 precision ---------
        fp_in_baby_imgs += len(unmatched_pr)

    # ---------- Close PR lists to avoid empty-class issues ----------
    ensure_present_for_all_classes()

    # ---------- AP/mAP (strict per class) ----------
    mAP, APs = compute_map_and_pr(per_true, per_score)

    # ---------- PR curves per class ----------
    pr_fig = plot_precision_recall(per_true, per_score, LABELS_MAP, mAP=mAP)
    pr_fig.savefig(figs_dir / "precision_recall.png", dpi=150, bbox_inches="tight")
    plt.close(pr_fig)

    # ---------- Confusion Matrices (with BG) ----------
    cm_figs = plot_confusion_matrix(y_true=y_true, y_pred=y_pred, labels_map=LABELS_MAP)
    cm_figs["raw"].savefig(figs_dir / "class_cm_raw.png", dpi=150, bbox_inches="tight")
    plt.close(cm_figs["raw"])
    cm_figs["normalized"].savefig(
        figs_dir / "class_cm_normalized.png", dpi=150, bbox_inches="tight"
    )
    plt.close(cm_figs["normalized"])

    # ---------- Boxplots (TP-only) ----------
    iou_data_tp = [
        {"class": LABELS_MAP[c], "iou": v} for c, vs in iou_errs.items() for v in vs
    ]
    if iou_data_tp:
        fig = plot_boxplots(
            iou_data_tp, "class", "iou", "IoU per Class", LABELS_MAP, y_lim=(0, 1)
        )
        fig.savefig(figs_dir / "iou_boxplot_tp.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    ang_data_tp = [
        {"class": LABELS_MAP[c], "error°": v}
        for c, vs in angle_errs.items()
        for v in vs
    ]
    if ang_data_tp:
        fig = plot_boxplots(
            ang_data_tp,
            "class",
            "error°",
            "Angle error per Class",
            LABELS_MAP,
            y_lim=(0, 180),
        )
        fig.savefig(figs_dir / "angle_boxplot_tp.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---------- IoU histogram & boxplot (ALL GT) ----------
    if len(iou_all_gt_global) > 0:
        arr = np.asarray(iou_all_gt_global, dtype=np.float32)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(arr, bins=20, range=(0, 1), edgecolor="black")
        ax.set_xlabel("IoU")
        ax.set_ylabel("Count")
        ax.set_title("IoU over ALL GT")
        fig.tight_layout()
        fig.savefig(figs_dir / "iou_hist_all_gt.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        iou_all_stats = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "std": float(arr.std(ddof=0)),
        }
        # Class boxplots of IoU vs GT (anchored)
        iou_gt_data = [
            {"class": LABELS_MAP[c], "iou": v}
            for c, vs in iou_all_gt_per_cls.items()
            for v in vs
        ]
        if iou_gt_data:
            fig = plot_boxplots(
                iou_gt_data, "class", "iou", "IoU per Class", LABELS_MAP, y_lim=(0, 1)
            )
            fig.savefig(
                figs_dir / "iou_boxplot_all_gt.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
    else:
        iou_all_stats = {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "std": 0.0,
        }

    # ---------- F1 vs threshold (strict multi-class) ----------
    f1_fig = plot_f1_vs_threshold(all_gts, all_scores, all_preds, LABELS_MAP)
    f1_fig.savefig(figs_dir / "f1_threshold.png", dpi=150, bbox_inches="tight")
    plt.close(f1_fig)

    # ---------- Adult-SOTA comparable summary (face/no-face) ----------
    # AP face/no-face and PR(best)
    ap_face_map, ap_face_per = compute_map_and_pr(per_true_face, per_score_face)
    ap_face = ap_face_per[0]
    # Curva loc-only (umbral) en modo face/no-face (denominador = #GT bebé)
    GT_baby_total = int(
        sum((loc_tp_per_cls[c] + loc_fn_per_cls[c]) for c in LABELS_MAP)
    )
    loc_curves = compute_loc_curves_from_predictions(
        y_is_tp=per_true_face[0],
        y_scores=per_score_face[0],
        n_gt=GT_baby_total,
        n_steps=200,
    )
    best_th = loc_curves["best_th"]
    best_P = loc_curves["best_P"]
    best_R = loc_curves["best_R"]
    best_F1 = loc_curves["best_F1"]

    # Global recall micro/macro (GT-anchored per class)
    TP_baby_total = int(sum(loc_tp_per_cls.values()))
    recall_micro_baby = (TP_baby_total / GT_baby_total) if GT_baby_total > 0 else 0.0
    recalls_per_cls = {
        c: (loc_tp_per_cls[c] / (loc_tp_per_cls[c] + loc_fn_per_cls[c]))
        if (loc_tp_per_cls[c] + loc_fn_per_cls[c]) > 0
        else 0.0
        for c in LABELS_MAP
    }
    recalls_present = [
        recalls_per_cls[c]
        for c in LABELS_MAP
        if (loc_tp_per_cls[c] + loc_fn_per_cls[c]) > 0
    ]
    recall_macro_baby = (
        float(np.mean(recalls_present)) if len(recalls_present) > 0 else 0.0
    )
    # --- Summary A rows (GT/TP/FN/Recall) por clase ---
    summary_sota_rows = []
    for c in LABELS_MAP:
        GT_c = int(loc_tp_per_cls[c] + loc_fn_per_cls[c])
        TP_c = int(loc_tp_per_cls[c])
        FN_c = int(loc_fn_per_cls[c])
        rec_c = (TP_c / GT_c) if GT_c > 0 else 0.0
        summary_sota_rows.append([LABELS_MAP[c], GT_c, TP_c, FN_c, f"{rec_c:.4f}"])

    # Localization precision (class-agnostic) and S1 precision
    recall_face_localization = (
        (loc_tp_global / (loc_tp_global + loc_fn_global))
        if (loc_tp_global + loc_fn_global) > 0
        else 0.0
    )
    precision_face_localization = (
        (loc_tp_global / (loc_tp_global + fp_global_loc))
        if (loc_tp_global + fp_global_loc) > 0
        else 0.0
    )
    precision_s1 = (
        (TP_baby_total / (TP_baby_total + fp_in_baby_imgs))
        if (TP_baby_total + fp_in_baby_imgs) > 0
        else 0.0
    )
    f1_s1 = (
        (2 * precision_s1 * recall_micro_baby / (precision_s1 + recall_micro_baby))
        if (precision_s1 + recall_micro_baby) > 0
        else 0.0
    )

    # ---------- Multi-class strict P/R/F1 ----------
    per_class_metrics = {}
    sum_tp = sum(stats[c]["tp"] for c in LABELS_MAP)
    sum_fp = sum(stats[c]["fp"] for c in LABELS_MAP)
    sum_fn = sum(stats[c]["fn"] for c in LABELS_MAP)
    micro_prec = (sum_tp / (sum_tp + sum_fp)) if (sum_tp + sum_fp) > 0 else 0.0
    micro_rec = (sum_tp / (sum_tp + sum_fn)) if (sum_tp + sum_fn) > 0 else 0.0
    micro_f1 = (
        (2 * micro_prec * micro_rec / (micro_prec + micro_rec))
        if (micro_prec + micro_rec) > 0
        else 0.0
    )

    macro_prec, macro_rec, macro_f1 = 0.0, 0.0, 0.0
    valid_classes = 0
    for c in LABELS_MAP:
        tp, fp, fn = stats[c]["tp"], stats[c]["fp"], stats[c]["fn"]
        prec_c = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec_c = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1_c = (2 * prec_c * rec_c / (prec_c + rec_c)) if (prec_c + rec_c) > 0 else 0.0
        per_class_metrics[c] = {
            "precision": prec_c,
            "recall": rec_c,
            "f1": f1_c,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        if (tp + fp) > 0 or (tp + fn) > 0:
            macro_prec += prec_c
            macro_rec += rec_c
            macro_f1 += f1_c
            valid_classes += 1
    if valid_classes > 0:
        macro_prec /= valid_classes
        macro_rec /= valid_classes
        macro_f1 /= valid_classes

    # ---------- Console summaries ----------
    # Summary A — format like SOTA
    print("\n[RESULTS — SOTA vs Baby]")
    print(f"  AP (face/no-face): {ap_face:.4f}")
    print(
        f"  PR(best) face/no-face: th={best_th:.3f}  P={best_P:.3f}  R={best_R:.3f}  F1={best_F1:.3f}  (GT={GT_baby_total})"
    )
    print(
        f"  Recall (global): micro={recall_micro_baby:.3f}  |  macro={recall_macro_baby:.3f}"
    )
    print(
        f"  [S1] Precision={precision_s1:.3f}  Recall={recall_micro_baby:.3f}  F1={f1_s1:.3f}"
    )
    for c in LABELS_MAP:
        GT_c = loc_tp_per_cls[c] + loc_fn_per_cls[c]
        TP_c = loc_tp_per_cls[c]
        FN_c = loc_fn_per_cls[c]
        print(
            f"  {LABELS_MAP[c]:<15s}  GT:{GT_c:4d}  TP:{TP_c:4d}  FN:{FN_c:4d}  Recall:{recalls_per_cls[c]:.3f}"
        )
    print(
        f"  FP buckets -> baby:{fp_in_baby_imgs}  adult_only:{fp_in_adult_imgs}  bg:{fp_in_bg_imgs}  |  FP_global:{fp_global_loc}"
    )
    if iou_all_stats["count"] > 0:
        print(
            f"  IoU(all GT): mean={iou_all_stats['mean']:.3f}  median={iou_all_stats['median']:.3f}  "
            f"p25={iou_all_stats['p25']:.3f}  p75={iou_all_stats['p75']:.3f}  std={iou_all_stats['std']:.3f}"
        )
    # Angle global stats (TP-only, unchanged behavior)
    all_angles = [a for cls in LABELS_MAP for a in angle_errs[cls]]
    angle_global_stats = {}
    if len(all_angles) > 0:
        all_angles_np = np.asarray(all_angles, dtype=np.float32)
        angle_global_stats = {
            "mean": float(all_angles_np.mean()),
            "median": float(np.median(all_angles_np)),
            "p25": float(np.percentile(all_angles_np, 25)),
            "p75": float(np.percentile(all_angles_np, 75)),
            "std": float(all_angles_np.std(ddof=0)),
            "count": int(all_angles_np.size),
        }
    else:
        angle_global_stats = {
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "std": 0.0,
            "count": 0,
        }

    # Summary B — multi-class strict
    print("\n[SUMMARY B — OBB multi-class strict (YOLO-OBB vs RetinaBabyFace)]")
    print(f"  mAP: {mAP:.4f}")
    print(f"  Micro: P={micro_prec:.3f}  R={micro_rec:.3f}  F1={micro_f1:.3f}")
    print(f"  Macro: P={macro_prec:.3f}  R={macro_rec:.3f}  F1={macro_f1:.3f}")
    for c in LABELS_MAP:
        m = per_class_metrics[c]
        ap_c = APs.get(c, 0.0)
        ang_mu = np.mean(angle_errs[c]) if angle_errs[c] else 0.0
        ang_sd = np.std(angle_errs[c]) if angle_errs[c] else 0.0
        print(
            f"  {LABELS_MAP[c]:<15s}  TP:{m['tp']:4d}  FP:{m['fp']:4d}  FN:{m['fn']:4d}  "
            f"P:{m['precision']:.3f}  R:{m['recall']:.3f}  F1:{m['f1']:.3f}  AP:{ap_c:.3f}  "
            f"Angleμ±σ:{ang_mu:.2f}±{ang_sd:.2f}"
        )
    # One extra line with angle global already printed above.

    # ---------- CSV consolidated ----------
    csv_path = out_dir / "metrics_obb.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        # Multi-class (strict)
        w.writerow(["# --- MULTI-CLASS (strict) ---"])
        w.writerow(["mAP", f"{mAP:.4f}"])
        w.writerow(["micro_P", f"{micro_prec:.4f}"])
        w.writerow(["micro_R", f"{micro_rec:.4f}"])
        w.writerow(["micro_F1", f"{micro_f1:.4f}"])
        w.writerow(["macro_P", f"{macro_prec:.4f}"])
        w.writerow(["macro_R", f"{macro_rec:.4f}"])
        w.writerow(["macro_F1", f"{macro_f1:.4f}"])
        w.writerow([])
        w.writerow(
            [
                "class",
                "TP",
                "FP",
                "FN",
                "Precision",
                "Recall",
                "F1",
                "AP_PR",
                "Angle_mean_deg",
                "Angle_std_deg",
            ]
        )
        for c in LABELS_MAP:
            ap_c = APs.get(c, 0.0)
            m = per_class_metrics[c]
            w.writerow(
                [
                    LABELS_MAP[c],
                    m["tp"],
                    m["fp"],
                    m["fn"],
                    f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}",
                    f"{m['f1']:.4f}",
                    f"{ap_c:.4f}",
                    f"{(np.mean(angle_errs[c]) if angle_errs[c] else 0.0):.2f}",
                    f"{(np.std(angle_errs[c])  if angle_errs[c] else 0.0):.2f}",
                ]
            )
        w.writerow([])

        # Adult-SOTA comparable (face/no-face)
        w.writerow(["# --- ADULT-SOTA COMPARABLE (face/no-face) ---"])
        w.writerow(["AP_face", f"{ap_face:.4f}"])
        w.writerow(["PR_best_th", f"{best_th:.4f}"])
        w.writerow(["PR_best_P", f"{best_P:.4f}"])
        w.writerow(["PR_best_R", f"{best_R:.4f}"])
        w.writerow(["PR_best_F1", f"{best_F1:.4f}"])
        w.writerow(["GT_total_babies", GT_baby_total])
        w.writerow(["recall_micro", f"{recall_micro_baby:.4f}"])
        w.writerow(["recall_macro", f"{recall_macro_baby:.4f}"])
        w.writerow(["precision_S1", f"{precision_s1:.4f}"])
        w.writerow(["f1_S1", f"{f1_s1:.4f}"])
        w.writerow(["FP_in_baby_imgs", fp_in_baby_imgs])
        w.writerow(["FP_in_adult_imgs", fp_in_adult_imgs])
        w.writerow(["FP_in_bg_imgs", fp_in_bg_imgs])
        w.writerow(["FP_global", fp_global_loc])
        w.writerow(["BG_instances_adult_total", bg_instances_adult_total])
        w.writerow(["BG_instances_pure_total", bg_instances_pure_total])
        w.writerow(
            ["BG_instances_total", bg_instances_adult_total + bg_instances_pure_total]
        )
        w.writerow([])
        # Per-class (Summary A style): GT / TP / FN / Recall
        w.writerow(["# --- SUMMARY A per-class (GT/TP/FN/Recall) ---"])
        w.writerow(["Class", "GT", "TP", "FN", "Recall"])
        w.writerows(summary_sota_rows)
        w.writerow([])
        # IoU over ALL GT
        w.writerow(["# --- IoU over ALL GT (GT-anchored) ---"])
        w.writerow(["IoU_count", iou_all_stats["count"]])
        w.writerow(["IoU_mean", f"{iou_all_stats['mean']:.4f}"])
        w.writerow(["IoU_median", f"{iou_all_stats['median']:.4f}"])
        w.writerow(["IoU_p25", f"{iou_all_stats['p25']:.4f}"])
        w.writerow(["IoU_p75", f"{iou_all_stats['p75']:.4f}"])
        w.writerow(["IoU_std", f"{iou_all_stats['std']:.4f}"])
        w.writerow([])
        # Angle global (TP-only)
        w.writerow(["# --- Angle (TP-only) global stats [deg] ---"])
        w.writerow(["Angle_count", angle_global_stats["count"]])
        w.writerow(["Angle_mean", f"{angle_global_stats['mean']:.4f}"])
        w.writerow(["Angle_median", f"{angle_global_stats['median']:.4f}"])
        w.writerow(["Angle_p25", f"{angle_global_stats['p25']:.4f}"])
        w.writerow(["Angle_p75", f"{angle_global_stats['p75']:.4f}"])
        w.writerow(["Angle_std", f"{angle_global_stats['std']:.4f}"])

    print(f"[INFO] Wrote consolidated metrics to {csv_path}")

    # ---------- Return everything useful ----------
    return {
        # Multi-class strict
        "mAP": mAP,
        "APs": APs,
        "stats": stats,
        "per_class_metrics": per_class_metrics,
        "micro_P": micro_prec,
        "micro_R": micro_rec,
        "micro_F1": micro_f1,
        "macro_P": macro_prec,
        "macro_R": macro_rec,
        "macro_F1": macro_f1,
        # Adult-SOTA comparable
        "AP_face": ap_face,
        "best_th_face": best_th,
        "best_P_face": best_P,
        "best_R_face": best_R,
        "best_F1_face": best_F1,
        "recall_face_localization": recall_face_localization,
        "precision_face_localization": precision_face_localization,
        "precision_S1": precision_s1,
        "recall_S1": recall_micro_baby,
        "f1_S1": f1_s1,
        "fp_in_baby_imgs": fp_in_baby_imgs,
        "fp_in_adult_imgs": fp_in_adult_imgs,
        "fp_in_bg_imgs": fp_in_bg_imgs,
        "fp_global_loc": fp_global_loc,
        "recalls_per_cls": recalls_per_cls,
        # IoU / Angle
        "iou_tp_only": iou_errs,
        "angle_tp_only": angle_errs,
        "iou_all_gt_per_cls": iou_all_gt_per_cls,
        "iou_all_gt_global_stats": iou_all_stats,
        "bg_instances_adult_total": bg_instances_adult_total,
        "bg_instances_pure_total": bg_instances_pure_total,
        "bg_instances_total": bg_instances_adult_total + bg_instances_pure_total,
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
        "--obb",
        action="store_true",
        help="Whether to evaluate YOLO-based oriented model",
    )
    ap.add_argument(
        "--model_version",
        type=str,
        help="Model variant: 'yolo' or 'retina' (for --obb only)",
    )
    ap.add_argument(
        "--aabb_mode",
        action="store_true",
        help="Whether to evaluate in AABB mode (for SOTA models only)",
    )
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching")
    ap.add_argument(
        "--min_score", type=float, default=0.0, help="Filter predictions by score"
    )
    args = ap.parse_args()

    if args.obb:
        if not args.sota_dir:
            raise ValueError(
                "For --obb evaluation, --sota-dir (predictions) is required."
            )
        evaluate_obb(
            data_root=Path(args.data_root),
            split=args.split,
            pred_dir=Path(args.sota_dir),
            out_dir=Path(args.output_dir),
            iou_th=args.iou,
            min_score=args.min_score,
            model_version=args.model_version,
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
