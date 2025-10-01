# evaluate_sota.py
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Tuple, Dict

from PIL import Image
import matplotlib.pyplot as plt

from benchmark.benchmark import (
    read_sota_preds_xywhr_xyxy,
    greedy_match,
    read_gt_baby_xywhr,
)
from engine.inference import plot_precision_recall, compute_map_and_pr

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
    Recorre test/, hace matching, computa:
      - PR/AP global cara/no-cara (SOTA)
      - Recall por orientación
      - TP/FP/FN globales y por clase
    Guarda plots y CSV.
    """
    images_dir = data_root / split / "images"
    labels_dir = data_root / split / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Acumuladores
    # PR global (face/no-face): un solo "clase" 0
    per_true_face = {0: []}
    per_score_face = {0: []}

    # Contadores por orientación
    gt_per_cls = {c: 0 for c in LABELS_MAP}
    tp_per_cls = {c: 0 for c in LABELS_MAP}
    fn_per_cls = {c: 0 for c in LABELS_MAP}
    fp_global = 0

    # Para lectura tamaño imagen
    def img_size(p: Path) -> Tuple[int, int]:
        with Image.open(p) as im:
            return im.size  # (W,H)

    # Loop por imágenes
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    jpgs = []
    for pat in exts:
        jpgs += list(images_dir.glob(pat))
    jpgs = sorted(jpgs)
    for img_p in jpgs:
        stem = img_p.stem
        gt_p = labels_dir / f"{stem}.txt"
        pr_p = sota_dir / f"{stem}.txt"

        W, H = img_size(img_p)

        # GT bebés
        gt_xywhr, gt_cls = read_gt_baby_xywhr(gt_p, (W, H))
        for c in gt_cls.tolist():
            gt_per_cls[int(c)] += 1

        # Pred SOTA
        pr_xywhr, pr_scores = read_sota_preds_xywhr_xyxy(
            pred_txt_path=pr_p, img_wh=(W, H), min_score=min_score
        )

        # Si no hay GT bebés:
        if gt_xywhr.numel() == 0:
            # Todas las predicciones son FP globales
            fp_global += int(pr_xywhr.shape[0])
            # También alimentar PR global (negativos)
            for s in pr_scores.tolist():
                per_true_face[0].append(0)
                per_score_face[0].append(float(s))
            continue

        # Matching
        matches, unmatched_gt, unmatched_pr = greedy_match(
            gt_xywhr, pr_xywhr, pr_scores, iou_th=iou_th
        )

        # Marcar PR global
        matched_pr_idx = set([m for (_, m, _) in matches])
        # alimentar PR global
        for j, s in enumerate(pr_scores.tolist()):
            is_tp = 1 if j in matched_pr_idx else 0
            per_true_face[0].append(is_tp)
            per_score_face[0].append(float(s))

        # contadores por clase (TP/FN) usando gt_cls
        matched_gt_idx = set([g for (g, _, _) in matches])
        for gi in matched_gt_idx:
            c = int(gt_cls[gi].item())
            tp_per_cls[c] += 1
        for gi in unmatched_gt:
            c = int(gt_cls[gi].item())
            fn_per_cls[c] += 1
    # ======= Métricas y salidas =======
    # AP global (cara/no-cara)
    mAP, APs = compute_map_and_pr(per_true_face, per_score_face)
    ap_face = APs[0]

    # Plot PR global
    pr_fig = plot_precision_recall(
        per_true_face, per_score_face, labels_map={0: "Face"}, mAP=mAP
    )
    pr_fig.savefig(out_dir / "precision_recall_face.png", dpi=150, bbox_inches="tight")
    plt.close(pr_fig)

    # Recall por orientación
    recalls = {
        c: (tp_per_cls[c] / gt_per_cls[c]) if gt_per_cls[c] > 0 else 0.0
        for c in LABELS_MAP
    }
    # Plot bar
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = list(LABELS_MAP.keys())
    ax.bar([LABELS_MAP[x] for x in xs], [recalls[x] for x in xs])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title("Recall por orientación (SOTA vs. GT bebé)")
    for i, c in enumerate(xs):
        ax.text(i, recalls[c] + 0.02, f"{recalls[c]:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "recall_per_orientation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # CSV
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

    print("\n[RESULTS]")
    print(f"  AP (face/no-face): {ap_face:.4f}")
    for c in LABELS_MAP:
        print(
            f"  {LABELS_MAP[c]:<15s}  GT:{gt_per_cls[c]:4d}  TP:{tp_per_cls[c]:4d}  "
            f"FN:{fn_per_cls[c]:4d}  Recall:{recalls[c]:.3f}"
        )
    print(f"  FP global (todas las imágenes): {fp_global}")

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


def main():
    ap = argparse.ArgumentParser(
        "Evaluate SOTA (face/no-face) contra GT bebé + orientaciones"
    )
    ap.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Raíz del dataset (con test/images y test/labels)",
    )
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument(
        "--sota-dir",
        type=str,
        required=True,
        help="Directorio con .txt de predicciones SOTA",
    )
    ap.add_argument("--out", type=str, required=True, help="Directorio de salida")
    ap.add_argument(
        "--iou", type=float, default=0.5, help="IoU threshold para matching"
    )
    ap.add_argument(
        "--min-score", type=float, default=0.0, help="Filtrar predicciones por score"
    )
    args = ap.parse_args()

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
