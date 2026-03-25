import math
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union, Optional, Callable

import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
)

from engine.train import (
    infer_with_rotated_nms,
    get_resize_size,
    generate_anchors_for_training,
    xyxyxyxy2xywhr,
    batch_probiou,
)

from utils.helpers import (
    to_numpy,
    ensure_polygons_42_shape,
    resolve_image_path,
)
from utils.visualize import (
    draw_predictions_on_image,
    plot_error_bar_mean_std_by_gt_bins_per_class,
    write_predictions_txt,
    xywhr_to_poly42_shape,
    scale_polys,
    scale_xywhr_boxes,
    get_oriented_face_crop,
    denormalize_image,
    plot_boxplots,
    plot_child_confusion_matrix,
    plot_confusion_matrix,
    plot_f1_vs_threshold,
    plot_precision_recall,
    plot_qualitative_grid,
    plot_error_bar_mean_std_by_gt_bins,
    plot_error_box_by_gt_bins,
)
from data_setup.augmentations import wrap_to_pi

# -----------------------------------------------------------------------------
# I. Model Checkpoint and Anchor Preparation
# -----------------------------------------------------------------------------


def load_model_checkpoint(
    model: torch.nn.Module, path: str, device: torch.device
) -> None:
    """
    Loads the model weights from a checkpoint file and prepares it for inference.

    Args:
        model (torch.nn.Module): The model to load the weights into.
        path (str): Path to the checkpoint file (.pth or .pt).
        device (torch.device): The device to load the model onto.
    """
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get(
        "model_state_dict", checkpoint
    )  # support plain or wrapped checkpoints
    model.load_state_dict(state_dict)
    model.to(device).eval()  # set model to evaluation mode
    logging.info(f"Model checkpoint loaded from {path}")


def prepare_anchors(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    scale_factors: List[float],
    ratio_factors: List[float],
    anchors_cache_path: str,
) -> Tuple[Tuple[int, int], torch.Tensor, torch.Tensor]:
    """
    Prepares anchors for inference using the base OBB statistics and given scale/ratio factors.

    Args:
        model (torch.nn.Module): The model used for feature extraction to generate anchors.
        loader (DataLoader): DataLoader to estimate image resize size.
        device (torch.device): Device for tensor creation (usually 'cuda' or 'cpu').
        scale_factors (List[float]): List of scaling factors for anchor generation.
        ratio_factors (List[float]): List of aspect ratio factors for anchors.
        anchors_cache_path (str): Path to cache the generated anchors for reuse.
    Returns:
        Tuple containing:
            - resize_size (Tuple[int, int]): Target size used to resize images.
            - anchors_xy (torch.Tensor): Anchors in (x, y) format for initial location.
            - anchors_xywhr (torch.Tensor): Anchors in (x, y, w, h, θ) format.
    """
    # Get target resize size from the dataset
    resize_size = get_resize_size(loader)

    # Generate anchors in both formats
    anchors_xy, anchors_xywhr = generate_anchors_for_training(
        model=model,
        resize_size=resize_size,
        device=device,
        scale_factors=scale_factors,
        ratio_factors=ratio_factors,
        anchors_cache_path=anchors_cache_path,
    )

    print(
        f"[INFO] Generated {anchors_xy.shape[0]} anchors for image size {resize_size}"
    )
    logging.info(f"✅ Anchors prepared successfully")
    return resize_size, anchors_xy, anchors_xywhr


# -----------------------------------------------------------------------------
# II. Inference Loop & Data Accumulation
# -----------------------------------------------------------------------------


def run_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    anchors_xy: torch.Tensor,
    resize_size: Tuple[int, int],
    face_thres: float,
    iou_thres: float,
    class_thres: float,
    baby_thres: float,
    device: torch.device,
    labels_map: Dict[int, str],
    render_original: bool = False,
) -> Dict[str, Any]:
    """
    Run inference and compute:
      - Orientation multi-class metrics (classes in labels_map + BG=-1 in CM accounting)
      - Child/adult binary metrics
      - Localization-only (SOTA comparable) using ONLY final detections

    Expected keys per sample from infer_with_rotated_nms:
        - boxes: Tensor [N, 5] post-NMS candidate boxes (xywhr)
        - labels: Tensor [N] predicted orientation labels
        - child_score: Tensor [N] baby probabilities
        - final_keep: Bool Tensor [N] selecting final detections
        - final_score: Tensor [N] score used to rank final detections
    """
    # -------------------------------------------------------------------------
    # Orientation metrics
    # -------------------------------------------------------------------------
    per_true: Dict[int, List[int]] = {c: [] for c in labels_map}
    per_score: Dict[int, List[float]] = {c: [] for c in labels_map}
    stats: Dict[int, Dict[str, int]] = {
        c: {"tp": 0, "fp": 0, "fn": 0} for c in labels_map
    }

    y_true: List[int] = []
    y_pred: List[int] = []

    iou_errs: Dict[int, List[float]] = {c: [] for c in labels_map}
    angle_errs: Dict[int, List[float]] = {c: [] for c in labels_map}

    bin_degs = (20, 10, 5)
    angle_errs_by_gtbin_global: Dict[int, Dict[int, List[float]]] = {
        bd: {} for bd in bin_degs
    }
    angle_errs_by_gtbin_per_cls: Dict[int, Dict[int, Dict[int, List[float]]]] = {
        bd: {c: {} for c in labels_map} for bd in bin_degs
    }

    def gt_bin_index(theta_deg_unsigned: float, bin_deg: int) -> int:
        """
        Return bin index for an unsigned GT angle in [0, 180).
        """
        theta = min(theta_deg_unsigned, 180.0 - 1e-6)
        return int(theta // bin_deg)

    # -------------------------------------------------------------------------
    # Child/adult metrics
    # -------------------------------------------------------------------------
    child_stats = {"tp": 0, "fp": 0, "fn": 0}
    child_gt: List[int] = []
    child_pred: List[int] = []

    def log_child(gt_is_baby: bool, pred_is_baby: bool) -> None:
        """
        Append labels for child/adult confusion matrix.
        """
        child_gt.append(1 if gt_is_baby else 0)
        child_pred.append(1 if pred_is_baby else 0)

    # -------------------------------------------------------------------------
    # Shared accumulators
    # -------------------------------------------------------------------------
    samples: List[Any] = []
    all_gts: List[int] = []
    all_preds: List[int] = []
    all_scores: List[float] = []

    # -------------------------------------------------------------------------
    # Localization-only (SOTA comparable) from final detections only
    # -------------------------------------------------------------------------
    loc_tp_global = 0
    loc_fp_global = 0
    loc_fn_global = 0

    loc_tp_per_cls = {c: 0 for c in labels_map}
    loc_fn_per_cls = {c: 0 for c in labels_map}
    loc_tp_pred_cls = {c: 0 for c in labels_map}
    loc_fp_pred_cls = {c: 0 for c in labels_map}

    fp_in_baby_imgs = 0
    fp_in_adult_imgs = 0
    fp_in_bg_imgs = 0

    distance_rows: List[Dict[str, Any]] = []

    dataset = loader.dataset
    global_idx = 0

    # Useful class tensor (fixed on device)
    orient_class_ids_device = torch.tensor(
        list(labels_map.keys()), device=device, dtype=torch.long
    )

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference"):
            imgs = batch["image"].to(device)
            targets = batch["target"]

            outputs = infer_with_rotated_nms(
                model,
                imgs,
                anchors_xy,
                resize_size,
                face_thres=face_thres,
                baby_thres=baby_thres,
                iou_thres=iou_thres,
                class_thres=class_thres,
            )

            batch_size = imgs.size(0)
            for b in range(batch_size):
                fname = dataset.file_list[global_idx]
                global_idx += 1
                fp_img, fn_img = 0, 0

                viz_payload = None
                if render_original:
                    orig_img_np, (sx, sy) = load_original_and_scale(
                        dataset, fname, resize_size
                    )
                    if orig_img_np is not None:
                        viz_payload = {"orig_img": orig_img_np, "scale": (sx, sy)}

                # -----------------------------------------------------------------
                # GT prep: index on CPU first, then move to device
                # -----------------------------------------------------------------
                valid_mask_cpu = targets["valid_mask"][b].bool().cpu()
                gt_boxes = targets["boxes"][b][valid_mask_cpu].to(device)
                gt_angles = targets["angles"][b][valid_mask_cpu].view(-1).to(device)
                gt_labels = targets["class_idx"][b][valid_mask_cpu].to(device)
                gt_child = targets["child_prob"][b][valid_mask_cpu].to(device) > 0.5
                num_gt = int(gt_boxes.size(0))

                is_bg_pure = num_gt == 0
                is_adult_only = (num_gt > 0) and (not bool(gt_child.any().item()))

                if num_gt > 0:
                    gt_xywhr = xyxyxyxy2xywhr(
                        gt_boxes, gt_angles.unsqueeze(-1), resize_size
                    ).to(device)
                else:
                    gt_xywhr = torch.empty((0, 5), device=device)

                gt_matched_any = torch.zeros(num_gt, dtype=torch.bool, device=device)
                gt_matched_orient = torch.zeros(num_gt, dtype=torch.bool, device=device)

                if num_gt > 0:
                    gt_class_in_eval = (
                        gt_labels.unsqueeze(1) == orient_class_ids_device.unsqueeze(0)
                    ).any(dim=1)
                    gt_is_orient_eval = gt_child & gt_class_in_eval
                else:
                    gt_is_orient_eval = torch.zeros(0, dtype=torch.bool, device=device)

                # -----------------------------------------------------------------
                # Predictions
                # -----------------------------------------------------------------
                pred_boxes_all = outputs[b]["boxes"].to(device)
                pred_labels_all = outputs[b]["labels"].to(device)
                pred_child_s_all = outputs[b]["child_score"].to(device)

                final_keep = outputs[b]["final_keep"].to(device).bool()
                pred_boxes = pred_boxes_all[final_keep]
                pred_labels = pred_labels_all[final_keep]
                pred_scores = outputs[b]["final_score"].to(device)[final_keep]

                num_pred_all = int(pred_boxes_all.size(0))
                num_pred = int(pred_boxes.size(0))

                # -----------------------------------------------------------------
                # Loc-only from final detections
                # -----------------------------------------------------------------
                loc_img = compute_loc_only_from_final(
                    gt_xywhr=gt_xywhr,
                    gt_labels=gt_labels,
                    pred_boxes_final=pred_boxes,
                    pred_scores_final=pred_scores,
                    pred_labels_final=pred_labels,
                    iou_threshold=iou_thres,
                    labels_map=labels_map,
                    device=device,
                    orient_class_ids_device=orient_class_ids_device,
                )

                loc_tp_global += loc_img["tp"]
                loc_fp_global += loc_img["fp"]
                loc_fn_global += loc_img["fn"]

                for c in labels_map:
                    loc_tp_per_cls[c] += loc_img["tp_per_cls"][c]
                    loc_fn_per_cls[c] += loc_img["fn_per_cls"][c]
                    loc_tp_pred_cls[c] += loc_img["tp_pred_cls"][c]
                    loc_fp_pred_cls[c] += loc_img["fp_pred_cls"][c]

                if is_bg_pure:
                    fp_in_bg_imgs += loc_img["fp"]
                elif is_adult_only:
                    fp_in_adult_imgs += loc_img["fp"]
                else:
                    fp_in_baby_imgs += loc_img["fp"]

                # detailed report rows (localization lens)
                for gi, pj in loc_img["matched_pairs"]:
                    iou_val = float(
                        batch_probiou(
                            gt_xywhr[gi].unsqueeze(0), pred_boxes[pj].unsqueeze(0)
                        ).item()
                    )
                    c_gt = int(gt_labels[gi].item())
                    c_pred = int(pred_labels[pj].item())
                    class_distance = (
                        abs(c_gt - c_pred)
                        if (c_gt in labels_map and c_pred in labels_map)
                        else None
                    )
                    distance_rows.append(
                        {
                            "image_id": str(fname),
                            "case": "loc_tp",
                            "gt_idx": gi,
                            "pred_idx": pj,
                            "gt_class": c_gt,
                            "pred_class": c_pred,
                            "class_distance": class_distance,
                            "iou": iou_val,
                            "score": float(pred_scores[pj].item()),
                        }
                    )

                for gi in loc_img["unmatched_gt"]:
                    c_gt = int(gt_labels[gi].item())
                    distance_rows.append(
                        {
                            "image_id": str(fname),
                            "case": "loc_fn",
                            "gt_idx": gi,
                            "pred_idx": None,
                            "gt_class": c_gt,
                            "pred_class": -1,
                            "class_distance": None,
                            "iou": None,
                            "score": 0.0,
                        }
                    )

                for pj in loc_img["unmatched_pr"]:
                    c_pred = int(pred_labels[pj].item())
                    distance_rows.append(
                        {
                            "image_id": str(fname),
                            "case": "loc_fp",
                            "gt_idx": None,
                            "pred_idx": pj,
                            "gt_class": -1,
                            "pred_class": c_pred,
                            "class_distance": None,
                            "iou": None,
                            "score": float(pred_scores[pj].item()),
                        }
                    )

                # -----------------------------------------------------------------
                # BG pure path for class/child metrics
                # -----------------------------------------------------------------
                if num_gt == 0:
                    # orientation/class CM: each final detection => BG -> class
                    if num_pred == 0:
                        # explicit BG->BG TN event
                        y_true.append(-1)
                        y_pred.append(-1)
                    else:
                        for det_idx in range(num_pred):
                            cls_det = int(pred_labels[det_idx].item())
                            score_det = float(pred_scores[det_idx].item())

                            if cls_det in labels_map:
                                per_true[cls_det].append(0)
                                per_score[cls_det].append(score_det)
                                stats[cls_det]["fp"] += 1
                                fp_img += 1

                            y_true.append(-1)
                            y_pred.append(cls_det)
                            all_gts.append(-1)
                            all_preds.append(cls_det)
                            all_scores.append(score_det)

                    samples.append(
                        (
                            imgs[b].cpu(),
                            {k: v.cpu().detach() for k, v in outputs[b].items()},
                            fname,
                            gt_boxes.cpu(),
                            gt_angles.cpu(),
                            gt_labels.cpu(),
                            fp_img,
                            fn_img,
                            viz_payload,
                        )
                    )
                    continue

                # -----------------------------------------------------------------
                # IoU matrices for class/child logic
                # -----------------------------------------------------------------
                iou_all = (
                    batch_probiou(gt_xywhr, pred_boxes_all)
                    if num_pred_all > 0
                    else torch.empty((num_gt, 0), device=device)
                )
                iou_final = (
                    batch_probiou(gt_xywhr, pred_boxes)
                    if num_pred > 0
                    else torch.empty((num_gt, 0), device=device)
                )

                # -----------------------------------------------------------------
                # A) ALL candidates for child/adult + gt_matched_any
                # -----------------------------------------------------------------
                if num_gt > 0:
                    pred_child_final = (
                        pred_child_s_all[final_keep]
                        if num_pred > 0
                        else torch.empty(0, device=device)
                    )

                    gt_matched_child = torch.zeros(
                        num_gt, dtype=torch.bool, device=device
                    )

                    if num_pred > 0:
                        _, det_order_child = torch.sort(pred_scores, descending=True)

                        for det_idx in det_order_child.tolist():
                            unmatched_gt = ~gt_matched_child
                            if not unmatched_gt.any():
                                break

                            ious_col = iou_final[:, det_idx].clone()
                            ious_col[~unmatched_gt] = -1.0

                            best_iou_val, best_gt_idx = ious_col.max(0)
                            best_iou_val = float(best_iou_val.item())
                            best_gt_idx = int(best_gt_idx.item())

                            if best_iou_val < iou_thres:
                                continue

                            gt_matched_child[best_gt_idx] = True

                            gt_is_baby = bool(gt_child[best_gt_idx].item())
                            pred_is_baby = bool(
                                pred_child_final[det_idx].item() >= baby_thres
                            )

                            log_child(gt_is_baby, pred_is_baby)

                            if gt_is_baby and pred_is_baby:
                                child_stats["tp"] += 1
                            elif (not gt_is_baby) and pred_is_baby:
                                child_stats["fp"] += 1
                            elif gt_is_baby and (not pred_is_baby):
                                child_stats["fn"] += 1

                    for gi in range(num_gt):
                        if bool(gt_matched_child[gi].item()):
                            continue

                        gt_is_baby = bool(gt_child[gi].item())
                        log_child(gt_is_baby, False)

                        if gt_is_baby:
                            child_stats["fn"] += 1

                # -----------------------------------------------------------------
                # B) FINAL outputs for orientation class CM
                # -----------------------------------------------------------------
                if num_pred > 0:
                    _, det_order = torch.sort(pred_scores, descending=True)

                    for det_idx in det_order.tolist():
                        score_det = float(pred_scores[det_idx].item())
                        cls_det = int(pred_labels[det_idx].item())

                        eligible = gt_is_orient_eval & (~gt_matched_orient)

                        if not eligible.any():
                            if cls_det in labels_map:
                                stats[cls_det]["fp"] += 1
                                fp_img += 1
                                per_true[cls_det].append(0)
                                per_score[cls_det].append(score_det)

                            y_true.append(-1)
                            y_pred.append(cls_det)
                            all_gts.append(-1)
                            all_preds.append(cls_det)
                            all_scores.append(score_det)
                            continue

                        ious_col = iou_final[:, det_idx].clone()
                        ious_col[~eligible] = -1.0
                        best_iou_val, best_gt_idx = ious_col.max(0)
                        best_iou_val = float(best_iou_val.item())
                        best_gt_idx = int(best_gt_idx.item())

                        if best_iou_val < iou_thres:
                            if cls_det in labels_map:
                                stats[cls_det]["fp"] += 1
                                fp_img += 1
                                per_true[cls_det].append(0)
                                per_score[cls_det].append(score_det)

                            y_true.append(-1)
                            y_pred.append(cls_det)
                            all_gts.append(-1)
                            all_preds.append(cls_det)
                            all_scores.append(score_det)
                            continue

                        true_cls = int(gt_labels[best_gt_idx].item())
                        gt_matched_orient[best_gt_idx] = True

                        if true_cls in labels_map:
                            iou_errs[true_cls].append(best_iou_val)

                            angle_diff = pred_boxes[det_idx, 4] - gt_angles[best_gt_idx]
                            error_deg = float(
                                wrap_to_pi(angle_diff).abs() * 180.0 / math.pi
                            )
                            angle_errs[true_cls].append(error_deg)

                            theta_gt_rad = float(
                                wrap_to_pi(gt_angles[best_gt_idx]).item()
                            )
                            theta_gt_deg_unsigned = min(
                                abs(theta_gt_rad * 180.0 / math.pi),
                                180.0 - 1e-6,
                            )
                            for bd in bin_degs:
                                bidx = gt_bin_index(theta_gt_deg_unsigned, bd)
                                angle_errs_by_gtbin_global[bd].setdefault(
                                    bidx, []
                                ).append(error_deg)
                                angle_errs_by_gtbin_per_cls[bd][true_cls].setdefault(
                                    bidx, []
                                ).append(error_deg)

                        if cls_det == true_cls and true_cls in stats:
                            stats[true_cls]["tp"] += 1
                            per_true[true_cls].append(1)
                            per_score[true_cls].append(score_det)

                            y_true.append(true_cls)
                            y_pred.append(true_cls)

                            all_gts.append(true_cls)
                            all_preds.append(true_cls)
                            all_scores.append(score_det)

                        else:
                            if cls_det in stats:
                                stats[cls_det]["fp"] += 1
                                fp_img += 1
                                per_true[cls_det].append(0)
                                per_score[cls_det].append(score_det)

                            if true_cls in stats:
                                stats[true_cls]["fn"] += 1
                                fn_img += 1

                            y_true.append(true_cls)
                            y_pred.append(cls_det)
                            all_gts.append(true_cls)
                            all_preds.append(cls_det)
                            all_scores.append(score_det)

                # unmatched evaluable GT -> class FN
                for i in range(num_gt):
                    if not bool(gt_is_orient_eval[i].item()):
                        continue
                    if bool(gt_matched_orient[i].item()):
                        continue

                    cls_gt = int(gt_labels[i].item())
                    if cls_gt in labels_map:
                        per_true[cls_gt].append(1)
                        per_score[cls_gt].append(0.0)
                        stats[cls_gt]["fn"] += 1
                        fn_img += 1

                        y_true.append(cls_gt)
                        y_pred.append(-1)

                    all_gts.append(cls_gt)
                    all_preds.append(-1)
                    all_scores.append(0.0)

                # Adults into BG row for class CM
                if num_pred > 0:
                    adult_idx = (gt_labels == -1).nonzero(as_tuple=False).view(-1)
                    if adult_idx.numel() > 0:
                        adult_taken = torch.zeros(
                            adult_idx.numel(), dtype=torch.bool, device=device
                        )
                        _, det_order_adult = torch.sort(pred_scores, descending=True)

                        for det_idx in det_order_adult.tolist():
                            ious_adult = iou_final[adult_idx, det_idx].clone()
                            ious_adult[adult_taken] = -1.0
                            best_iou_val, best_local = ious_adult.max(0)
                            best_iou_val = float(best_iou_val.item())
                            best_local = int(best_local.item())

                            if best_iou_val < iou_thres:
                                continue

                            adult_taken[best_local] = True
                            cls_det = int(pred_labels[det_idx].item())
                            score_det = float(pred_scores[det_idx].item())

                            y_true.append(-1)
                            y_pred.append(cls_det)
                            all_gts.append(-1)
                            all_preds.append(cls_det)
                            all_scores.append(score_det)

                        for k in range(adult_idx.numel()):
                            if not bool(adult_taken[k].item()):
                                y_true.append(-1)
                                y_pred.append(-1)
                else:
                    adult_count = int((gt_labels == -1).sum().item())
                    for _ in range(adult_count):
                        y_true.append(-1)
                        y_pred.append(-1)

                samples.append(
                    (
                        imgs[b].cpu(),
                        {k: v.cpu().detach() for k, v in outputs[b].items()},
                        fname,
                        gt_boxes.cpu(),
                        gt_angles.cpu(),
                        gt_labels.cpu(),
                        fp_img,
                        fn_img,
                        viz_payload,
                    )
                )

    # non-empty PR vectors
    for cls in labels_map:
        if not per_true[cls]:
            per_true[cls].append(0)
            per_score[cls].append(0.0)

    # Loc-only summaries
    loc_precision = (
        loc_tp_global / (loc_tp_global + loc_fp_global)
        if (loc_tp_global + loc_fp_global) > 0
        else 0.0
    )
    loc_recall = (
        loc_tp_global / (loc_tp_global + loc_fn_global)
        if (loc_tp_global + loc_fn_global) > 0
        else 0.0
    )
    loc_f1 = (
        2 * loc_precision * loc_recall / (loc_precision + loc_recall)
        if (loc_precision + loc_recall) > 0
        else 0.0
    )

    precision_loc_pred_per_cls = {}
    recall_loc_per_cls = {}
    for c in labels_map:
        tp_c = loc_tp_per_cls[c]
        fn_c = loc_fn_per_cls[c]
        recall_loc_per_cls[c] = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0

        tp_pred_c = loc_tp_pred_cls[c]
        fp_pred_c = loc_fp_pred_cls[c]
        precision_loc_pred_per_cls[c] = (
            tp_pred_c / (tp_pred_c + fp_pred_c) if (tp_pred_c + fp_pred_c) > 0 else 0.0
        )

    print(f"[INFO] Inference completed on {global_idx} samples.")
    print(
        f"[LOC-ONLY] TP={loc_tp_global}, FP={loc_fp_global}, FN={loc_fn_global}, "
        f"P={loc_precision:.4f}, R={loc_recall:.4f}, F1={loc_f1:.4f}"
    )

    return {
        # orientation/class
        "per_true": per_true,
        "per_score": per_score,
        "iou_errs": iou_errs,
        "angle_errs": angle_errs,
        "stats": stats,
        "y_true": y_true,
        "y_pred": y_pred,
        "all_gts": all_gts,
        "all_preds": all_preds,
        "all_scores": all_scores,
        "angle_errs_by_gtbin_global": angle_errs_by_gtbin_global,
        "angle_errs_by_gtbin_per_cls": angle_errs_by_gtbin_per_cls,
        "bin_degs": bin_degs,
        # child/adult
        "child_stats": child_stats,
        "child_gt": child_gt,
        "child_pred": child_pred,
        # localization-only (from final detections)
        "loc_tp_global": loc_tp_global,
        "loc_fp_global": loc_fp_global,
        "loc_fn_global": loc_fn_global,
        "loc_precision": loc_precision,
        "loc_recall": loc_recall,
        "loc_f1": loc_f1,
        "loc_tp_per_cls": loc_tp_per_cls,
        "loc_fn_per_cls": loc_fn_per_cls,
        "loc_tp_pred_cls": loc_tp_pred_cls,
        "loc_fp_pred_cls": loc_fp_pred_cls,
        "precision_loc_pred_per_cls": precision_loc_pred_per_cls,
        "recall_loc_per_cls": recall_loc_per_cls,
        "fp_in_baby_imgs": fp_in_baby_imgs,
        "fp_in_adult_imgs": fp_in_adult_imgs,
        "fp_in_bg_imgs": fp_in_bg_imgs,
        "distance_rows": distance_rows,
        # qualitative
        "samples": samples,
    }


# -----------------------------------------------------------------------------
# III. Metric Computation & Plotting
# -----------------------------------------------------------------------------


def compute_map_and_pr(
    per_true: Dict[int, List[int]], per_score: Dict[int, List[float]]
) -> Tuple[float, Dict[int, float]]:
    """
    Computes the mean Average Precision (mAP) and per-class AP using precision-recall curves.

    Args:
        per_true (Dict[int, List[int]]): Binary ground truth (1 for TP, 0 for FN/FP) per class.
        per_score (Dict[int, List[float]]): Confidence scores of predictions per class.

    Returns:
        Tuple:
            - float: mean Average Precision across all classes.
            - Dict[int, float]: Average Precision per class.
    """
    APs = {
        cls: (
            average_precision_score(per_true[cls], per_score[cls])
            if sum(per_true[cls]) > 0
            else 0.0
        )
        for cls in per_true
    }
    mAP = float(np.mean(list(APs.values())))
    print(f"[INFO] Computed mAP: {mAP:.4f}")
    return mAP, APs


def save_individual_predictions(
    samples: List[
        Tuple[
            Any,
            Dict[str, torch.Tensor],
            str,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
            int,
            Optional[Dict[str, Any]],
        ]
    ],
    labels_map: Dict[int, str],
    output_dir: str,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    split_by_error: bool = True,
    viz_original_res: bool = False,
    orig_sizeresolver: Optional[Callable[[str], Optional[Tuple[int, int]]]] = None,
    resize_size: Tuple[int, int] = (640, 640),
) -> None:
    """
    Saves individual visualizations of predictions with ground truth for qualitative analysis.

    This function generates and saves visualization images showing both ground truth and predicted
    oriented bounding boxes (OBBs), with options to use original image resolution and split results
    by error type.

    Args:
        samples: List of tuples containing:
            - Image tensor (normalized)
            - Prediction dictionary with keys:
                - 'polygons': Predicted OBB vertices
                - 'labels': Predicted class labels
                - 'scores': Confidence scores
                - 'boxes': OBB parameters (x,y,w,h,θ)
            - File name
            - Ground truth OBB vertices
            - Ground truth angles
            - Ground truth labels
            - False positive count
            - False negative count
            - Optional visualization payload with:
                - 'orig_img': Original resolution image
                - 'scale': (sx,sy) scaling factors
        labels_map: Mapping from class indices to human-readable labels
        output_dir: Base directory for saving visualizations
        mean: Channel means for image denormalization
        std: Channel standard deviations for denormalization
        split_by_error: Whether to organize outputs into error type subdirectories:
            - tp_only/: Perfect predictions
            - fp/: False positives only
            - fn/: False negatives only
            - fp_fn/: Both error types
        viz_original_res: Whether to render at original image resolution
        orig_sizeresolver: Function to get original (W,H) from filename
        resize_size: Target size used during resizing/inference

    The visualization includes:
        - Ground truth OBBs in dashed green with orange front edge
        - Predicted OBBs in solid blue with red front edge
        - Class labels, angles and confidence scores
        - Optional background in original resolution

    Example directory structure when split_by_error=True:
        output_dir/
        ├── tp_only/
        │   ├── image1.jpg
        │   └── image2.jpg
        ├── fp/
        │   └── image3.jpg
        ├── fn/
        │   └── image4.jpg
        └── fp_fn/
            └── image5.jpg
    """
    # Convert output directory to Path object for easier manipulation
    base_dir = Path(output_dir)

    # Create subdirectories for different error types if splitting is enabled
    if split_by_error:
        for sub in ("tp_only", "fp", "fn", "fp_fn"):
            (base_dir / sub).mkdir(parents=True, exist_ok=True)
    else:
        base_dir.mkdir(parents=True, exist_ok=True)

    # Process each sample
    for sample in samples:
        # Handle samples with or without visualization payload
        if len(sample) == 9:
            img_t, out, fname, gt_b, gt_a, gt_l, fp_img, fn_img, viz = sample
        else:
            img_t, out, fname, gt_b, gt_a, gt_l, fp_img, fn_img = sample
            viz = None

        # Determine background image and scaling factors
        if viz is not None and viz.get("orig_img", None) is not None:
            # Use provided original resolution image if available
            base_img = viz["orig_img"]  # np.uint8 (H0, W0, 3)
            sx, sy = viz["scale"]
        else:
            # Fallback to denormalized tensor and try upscaling if requested
            base_img = denormalize_image(img_t, mean=mean, std=std)  # (Hr, Wr, 3)
            sx, sy = 1.0, 1.0
            if viz_original_res and orig_sizeresolver is not None:
                wh = orig_sizeresolver(fname)
                if wh is not None:
                    W0, H0 = wh
                    Wr, Hr = resize_size
                    sx = float(W0) / float(Wr)
                    sy = float(H0) / float(Hr)
                    try:
                        from PIL import Image

                        base_img = np.asarray(
                            Image.fromarray(base_img).resize((W0, H0))
                        )
                    except Exception:
                        sx, sy = 1.0, 1.0  # fallback to 640x640 if resize fails

        # Setup matplotlib figure to match image dimensions exactly
        H_out, W_out = int(base_img.shape[0]), int(base_img.shape[1])
        dpi = 100
        fig = plt.figure(figsize=(W_out / dpi, H_out / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  # use full canvas without margins

        # Display base image with correct alignment and no interpolation artifacts
        ax.imshow(base_img, extent=(0, W_out, H_out, 0), interpolation="nearest")
        ax.set_xlim(0, W_out)
        ax.set_ylim(H_out, 0)  # invert Y axis for image coordinates
        ax.axis("off")

        # Draw ground truth OBBs (dashed green with orange front edge)
        for pts, ang, lbl in zip(gt_b, gt_a, gt_l):
            coords = pts.detach().cpu().view(4, 2).numpy()
            # Scale coordinates to original resolution if needed
            coords[:, 0] *= sx
            coords[:, 1] *= sy
            # Draw OBB polygon
            ax.add_patch(
                patches.Polygon(
                    coords,
                    closed=True,
                    fill=False,
                    edgecolor="#008000",  # Dark green for GT
                    linewidth=2,
                    linestyle="--",
                )
            )
            # Draw front edge in orange
            ax.plot(coords[[0, 1], 0], coords[[0, 1], 1], color="orange", linewidth=2)

            # Add class label and angle at bottom-right with green background
            br_x, br_y = coords[:, 0].max(), coords[:, 1].max()
            ax.text(
                br_x,
                br_y,
                f"{labels_map.get(int(lbl), 'unknown')}: {math.degrees(float(ang)):.1f}°",
                color="white",
                fontsize=6,
                fontweight="bold",
                ha="right",
                va="bottom",
                bbox=dict(facecolor="#008000", alpha=0.8, edgecolor="none", pad=2.5),
            )

        # Draw predicted OBBs (solid blue with red front edge)
        final_keep = out.get("final_keep", None)

        if final_keep is not None:
            keep_mask = final_keep.bool().cpu()
            pred_polygons = out["polygons"][keep_mask]
            pred_labels = out["labels"][keep_mask]
            pred_scores = out["final_score"][keep_mask]
            pred_boxes = out["boxes"][keep_mask]
        else:
            pred_polygons = out["polygons"]
            pred_labels = out["labels"]
            pred_scores = out["final_score"]
            pred_boxes = out["boxes"]

        for i, (pts, lbl, score) in enumerate(
            zip(pred_polygons, pred_labels, pred_scores)
        ):
            coords = pts.cpu().view(4, 2).numpy()
            coords[:, 0] *= sx
            coords[:, 1] *= sy

            ax.add_patch(
                patches.Polygon(
                    coords,
                    closed=True,
                    fill=False,
                    edgecolor="#004080",
                    linewidth=1.5,
                )
            )

            ax.plot(
                coords[[0, 1], 0],
                coords[[0, 1], 1],
                color="#800000",
                linewidth=1.5,
            )

            tl_x, tl_y = coords[:, 0].min(), coords[:, 1].min()
            ang_pred = math.degrees(float(pred_boxes[i, 4]))

            ax.text(
                tl_x,
                tl_y,
                f"{labels_map.get(int(lbl), 'unknown')}: {ang_pred:.1f}° / {score:.2f}",
                color="white",
                fontsize=6,
                ha="left",
                va="top",
                bbox=dict(facecolor="#004080", alpha=0.9, edgecolor="none", pad=2.5),
            )

        # Determine output subdirectory based on error types
        if not split_by_error:
            save_dir = base_dir
        else:
            if fp_img and not fn_img:
                subdir = "fp"  # False positives only
            elif fn_img and not fp_img:
                subdir = "fn"  # False negatives only
            elif fp_img and fn_img:
                subdir = "fp_fn"  # Both false positives and false negatives
            else:
                subdir = "tp_only"  # Perfect predictions (true positives only)

            save_dir = base_dir / subdir
            save_dir.mkdir(exist_ok=True, parents=True)

        # Save visualization without padding and close to free memory
        fig.savefig(
            save_dir / Path(fname).name, dpi=dpi, bbox_inches=None, pad_inches=0
        )
        plt.close(fig)

    print(f"[INFO] Saved individual predictions to {output_dir}")


# -----------------------------------------------------------------------------
# V. Main Entry
# -----------------------------------------------------------------------------


def inference(
    model: torch.nn.Module,
    test_loader: DataLoader,
    output_dir: Union[str, Path],
    device: torch.device,
    labels_map: Dict[int, str],
    scale_factors: List[float],
    ratio_factors: List[float],
    face_thres: float = 0.25,
    baby_thres: float = 0.25,
    iou_thres: float = 0.5,
    class_thres: float = 0.5,
    grid_shape: Tuple[int, int] = (3, 3),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    save_figs: bool = True,
    close_figs: bool = True,
    anchors_cache_path: Union[str, Path] = None,
    render_original: bool = False,
) -> Dict[str, Any]:
    """
        This function processes a test dataset through a trained model and generates comprehensive
    evaluation metrics and visualizations.

    Steps:
        1. Anchor preparation for inference
        2. Model inference on test set
        3. Metrics computation and visualization generation
        4. CSV export of metrics and confusion matrices
        5. Saving of prediction visualizations

        - model (torch.nn.Module): Trained model for inference.
        - test_loader (DataLoader): DataLoader containing test dataset.
        - output_dir (Union[str, Path]): Directory path to save results and visualizations.
        - device (torch.device): Computing device ('cuda' or 'cpu').
        - labels_map (Dict[int, str]): Mapping of class indices to label names.
        - scale_factors (List[float]): Scale factors for anchor box generation.
        - ratio_factors (List[float]): Aspect ratio factors for anchor box generation.
        - face_thres (float, optional): Confidence threshold for face detection. Defaults to 0.25.
        - baby_thres (float, optional): Confidence threshold for baby classification. Defaults to 0.25.
        - iou_thres (float, optional): IoU threshold for prediction matching. Defaults to 0.5.
        - class_thres (float, optional): Confidence threshold for class predictions. Defaults to 0.5.
        - grid_shape (Tuple[int, int], optional): Shape of prediction visualization grid (rows, cols).
            Defaults to (3, 3).
        - mean (Tuple[float, float, float], optional): Mean values for image normalization.
            Defaults to (0.485, 0.456, 0.406).
        - std (Tuple[float, float, float], optional): Standard deviation values for image normalization.
            Defaults to (0.229, 0.224, 0.225).
        - save_figs (bool, optional): Whether to save generated figures. Defaults to True.
        - close_figs (bool, optional): Whether to close figures after saving. Defaults to True.
        - anchors_cache_path (Union[str, Path], optional): Path to cache generated anchors.
            Defaults to None.

            - "mAP": Mean Average Precision across all classes
            - "APs": Dictionary of per-class Average Precision scores
        -  render_original (bool, optional): Whether to render predictions on original images. Defaults to False.

    Generated Outputs:
        - Precision-Recall curves
        - Confusion matrices (raw and normalized) for class predictions
        - Confusion matrices (raw and normalized) for child/adult classification
        - IoU distribution boxplots per class
        - Angle error distribution boxplots per class
        - F1 score vs confidence threshold plots
        - Grid of qualitative prediction examples
        - Individual prediction visualizations
        - CSV files with metrics and confusion matrices
    """

    def save_figure(fig: plt.Figure, fname: str):
        """Helper to save a matplotlib figure if enabled."""
        if save_figs:
            fig.savefig(figures_dir / fname, dpi=150, bbox_inches="tight")
            if close_figs:
                plt.close(fig)

    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    predictions_dir = output_dir / "predictions"

    # Create output directories if they do not exist
    for d in (figures_dir, predictions_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("[STEP 1] Preparing anchors...")
    resize_size, anchors_xy, _ = prepare_anchors(
        model=model,
        loader=test_loader,
        device=device,
        scale_factors=scale_factors,
        ratio_factors=ratio_factors,
        anchors_cache_path=anchors_cache_path,
    )

    print("[STEP 2] Running inference...")
    results = run_evaluation(
        model=model,
        loader=test_loader,
        anchors_xy=anchors_xy,
        resize_size=resize_size,
        face_thres=face_thres,
        iou_thres=iou_thres,
        class_thres=class_thres,
        baby_thres=baby_thres,
        device=device,
        labels_map=labels_map,
        render_original=render_original,
    )

    print("[STEP 3] Computing metrics and plots...")
    mAP, APs = compute_map_and_pr(results["per_true"], results["per_score"])

    # Precision-Recall curve
    save_figure(
        plot_precision_recall(
            results["per_true"], results["per_score"], labels_map, mAP
        ),
        "precision_recall.png",
    )

    # Confusion matrices (raw and normalized)
    cm_figs = plot_child_confusion_matrix(
        y_true=results["child_gt"],
        y_pred=results["child_pred"],
    )
    save_figure(cm_figs["raw"], "child_cm_raw.png")
    save_figure(cm_figs["normalized"], "child_cm_normalized.png")

    # Confusion matrices (raw and normalized)
    cm_figs = plot_confusion_matrix(
        y_true=results["y_true"], y_pred=results["y_pred"], labels_map=labels_map
    )
    save_figure(cm_figs["raw"], "class_cm_raw.png")
    save_figure(cm_figs["normalized"], "class_cm_normalized.png")

    # IoU boxplots per class
    iou_data = [
        {"class": labels_map[c], "iou": v}
        for c, vals in results["iou_errs"].items()
        for v in vals
    ]
    save_figure(
        plot_boxplots(
            iou_data,
            "class",
            "iou",
            "IoU Distribution per Class",
            labels_map,
            y_lim=(0, 1),
        ),
        "iou_boxplot.png",
    )

    # Angle error boxplots per class
    ang_data = [
        {"class": labels_map[c], "error°": v}
        for c, vals in results["angle_errs"].items()
        for v in vals
    ]
    save_figure(
        plot_boxplots(
            ang_data,
            "class",
            "error°",
            "Angle-Error Distribution per Class",
            labels_map,
            y_lim=(0, 180),
        ),
        "angle_boxplot.png",
    )

    # Angular error by GT bins (global)
    for bd in results["bin_degs"]:
        buckets = results["angle_errs_by_gtbin_global"][bd]
        if buckets:
            fig_box_all, fig_box_filter = plot_error_box_by_gt_bins(
                buckets, bd, title="Angular error by GT angle bin"
            )
            save_figure(fig_box_all, f"box_angle_error_per_bin_filter_{bd}.png")
            save_figure(fig_box_filter, f"box_angle_error_per_bin_all_{bd}.png")

            fig_bar_all, fig_bar_filter = plot_error_bar_mean_std_by_gt_bins(
                buckets, bd, title="Angular error mean±std by GT angle bin"
            )
            save_figure(fig_bar_all, f"hist_angle_error_per_bin_filter_{bd}.png")
            save_figure(fig_bar_filter, f"hist_angle_error_per_bin_all_{bd}.png")

    # Per-class bars
    for bd in results["bin_degs"]:
        fig_bar_cls = plot_error_bar_mean_std_by_gt_bins_per_class(
            results["angle_errs_by_gtbin_per_cls"][bd],
            labels_map,
            bd,
            title_prefix="Angular error mean±std by GT angle bin per class",
        )
        save_figure(fig_bar_cls, f"angle_error_per_class_bar_bin_{bd}.png")

    # F1 score vs. confidence threshold
    save_figure(
        plot_f1_vs_threshold(
            results["all_gts"], results["all_scores"], results["all_preds"], labels_map
        ),
        "f1_threshold.png",
    )

    # Qualitative grid of predictions
    save_figure(
        plot_qualitative_grid(results["samples"], labels_map, grid_shape, mean, std),
        "grid_examples.png",
    )

    print("[STEP 3.1] Localization-only summary (SOTA comparable)...")
    print(
        "[LOC] "
        f"P={results['loc_precision']:.4f} | "
        f"R={results['loc_recall']:.4f} | "
        f"F1={results['loc_f1']:.4f}"
    )
    print(
        "[LOC] "
        f"TP={results['loc_tp_global']} | "
        f"FP={results['loc_fp_global']} | "
        f"FN={results['loc_fn_global']}"
    )
    print(
        "[LOC] FP buckets -> "
        f"BABY={results['fp_in_baby_imgs']} | "
        f"ADULT_ONLY={results['fp_in_adult_imgs']} | "
        f"BG={results['fp_in_bg_imgs']}"
    )

    print("[STEP 3.2] Exporting distance report...")
    distance_csv = export_distance_report(
        distance_rows=results["distance_rows"],
        out_dir=output_dir,
        fname="distance_report.csv",
    )

    print("[STEP 4] Exporting metrics and confusion matrix CSV...")
    metrics_csv = export_metrics_and_confusion_csv(results, labels_map, output_dir)

    print("[STEP 5] Saving individual prediction images...")
    resolver = build_image_sizeresolver(test_loader.dataset)
    save_individual_predictions(
        samples=results["samples"],
        labels_map=labels_map,
        output_dir=predictions_dir,
        mean=mean,
        std=std,
        split_by_error=True,
        viz_original_res=render_original,
        orig_sizeresolver=resolver,
        resize_size=resize_size,
    )
    print("[DONE] Inference and reporting completed.")

    return {
        "mAP": mAP,
        "APs": APs,
        "loc_precision": results["loc_precision"],
        "loc_recall": results["loc_recall"],
        "loc_f1": results["loc_f1"],
        "metrics_csv": metrics_csv,
        "distance_csv": distance_csv,
    }


def export_metrics_and_confusion_csv(
    results: dict,
    labels_map: Dict[int, str],
    out_dir: Path,
    fname: str = "metrics.csv",
) -> Path:
    """
    Export evaluation results to a single CSV with:
      1) Per-class metrics (orientation classes + BG row)
      2) Child/adult metrics summary
      3) Raw class confusion matrix
      4) Normalized class confusion matrix
      5) Raw child confusion matrix
      6) Normalized child confusion matrix

    Args:
        results: Output dictionary from run_evaluation.
        labels_map: Mapping from class id to class name for orientation classes.
        out_dir: Output directory.
        fname: CSV filename.

    Returns:
        Path to the saved CSV file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / fname

    classes = list(labels_map.keys())
    class_names = [labels_map[c] for c in classes]
    bg_label, bg_name = -1, "BG"

    # --------------------- Per-class metrics table ---------------------
    metric_rows = []
    for class_id, class_name in zip(classes, class_names):
        tp = int(results["stats"][class_id]["tp"])
        fp = int(results["stats"][class_id]["fp"])
        fn = int(results["stats"][class_id]["fn"])

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

        ap_pr = 0.0
        if len(results["per_true"][class_id]) > 0:
            ap_pr = float(
                average_precision_score(
                    results["per_true"][class_id], results["per_score"][class_id]
                )
            )

        iou_vals = results["iou_errs"][class_id]
        angle_vals = results["angle_errs"][class_id]

        metric_rows.append(
            {
                "Class": class_name,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "AP_PR": ap_pr,
                "IoU_mean": float(np.mean(iou_vals)) if iou_vals else 0.0,
                "IoU_std": float(np.std(iou_vals)) if iou_vals else 0.0,
                "Angle_mean_deg": float(np.mean(angle_vals)) if angle_vals else 0.0,
                "Angle_std_deg": float(np.std(angle_vals)) if angle_vals else 0.0,
            }
        )

    # BG summary row derived from class CM accounting
    y_true_np = np.array(results["y_true"])
    y_pred_np = np.array(results["y_pred"])

    bg_tp = int(((y_true_np == bg_label) & (y_pred_np == bg_label)).sum())
    bg_fn = int(((y_true_np == bg_label) & (y_pred_np != bg_label)).sum())
    bg_fp = int(((y_true_np != bg_label) & (y_pred_np == bg_label)).sum())

    bg_precision = bg_tp / (bg_tp + bg_fp) if (bg_tp + bg_fp) else 0.0
    bg_recall = bg_tp / (bg_tp + bg_fn) if (bg_tp + bg_fn) else 0.0
    bg_f1 = (
        2 * bg_tp / (2 * bg_tp + bg_fp + bg_fn) if (2 * bg_tp + bg_fp + bg_fn) else 0.0
    )

    metric_rows.append(
        {
            "Class": bg_name,
            "TP": bg_tp,
            "FP": bg_fp,
            "FN": bg_fn,
            "Precision": bg_precision,
            "Recall": bg_recall,
            "F1": bg_f1,
            "AP_PR": 0.0,
            "IoU_mean": 0.0,
            "IoU_std": 0.0,
            "Angle_mean_deg": 0.0,
            "Angle_std_deg": 0.0,
        }
    )

    df_metrics = pd.DataFrame(metric_rows).set_index("Class")

    # --------------------- Child metrics summary ---------------------
    child_tp = int(results["child_stats"]["tp"])
    child_fp = int(results["child_stats"]["fp"])
    child_fn = int(results["child_stats"]["fn"])

    child_precision = child_tp / (child_tp + child_fp) if (child_tp + child_fp) else 0.0
    child_recall = child_tp / (child_tp + child_fn) if (child_tp + child_fn) else 0.0
    child_f1 = (
        2 * child_tp / (2 * child_tp + child_fp + child_fn)
        if (2 * child_tp + child_fp + child_fn)
        else 0.0
    )

    df_child_metrics = pd.DataFrame(
        [
            {
                "Task": "Child_vs_Adult",
                "TP": child_tp,
                "FP": child_fp,
                "FN": child_fn,
                "Precision": child_precision,
                "Recall": child_recall,
                "F1": child_f1,
            }
        ]
    ).set_index("Task")

    # --------------------- Class confusion matrices ---------------------
    cm_labels = classes + [bg_label]
    cm_names = class_names + [bg_name]

    cm_raw = confusion_matrix(results["y_true"], results["y_pred"], labels=cm_labels)
    cm_norm = np.nan_to_num(cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True))

    df_cm_raw = pd.DataFrame(cm_raw, index=cm_names, columns=cm_names)
    df_cm_norm = pd.DataFrame(cm_norm, index=cm_names, columns=cm_names)

    # --------------------- Child confusion matrices ---------------------
    child_cm_raw = confusion_matrix(
        results["child_gt"], results["child_pred"], labels=[0, 1]
    )
    child_cm_norm = np.nan_to_num(
        child_cm_raw.astype(float) / child_cm_raw.sum(axis=1, keepdims=True)
    )
    df_child_cm_raw = pd.DataFrame(
        child_cm_raw,
        index=["Adult", "Child"],
        columns=["Adult", "Child"],
    )
    df_child_cm_norm = pd.DataFrame(
        child_cm_norm,
        index=["Adult", "Child"],
        columns=["Adult", "Child"],
    )

    # ---------- Localization-only (SOTA comparable) ----------
    loc_tp = int(results.get("loc_tp_global", 0))
    loc_fp = int(results.get("loc_fp_global", 0))
    loc_fn = int(results.get("loc_fn_global", 0))

    loc_p = float(results.get("loc_precision", 0.0))
    loc_r = float(results.get("loc_recall", 0.0))
    loc_f1 = float(results.get("loc_f1", 0.0))

    loc_tp_per_cls = results.get("loc_tp_per_cls", {})
    loc_fn_per_cls = results.get("loc_fn_per_cls", {})
    precision_loc_pred_per_cls = results.get("precision_loc_pred_per_cls", {})

    df_loc_per_cls = pd.DataFrame(
        [
            {
                "Class": labels_map[c],
                "GT_total": int(loc_tp_per_cls.get(c, 0) + loc_fn_per_cls.get(c, 0)),
                "TP_loc": int(loc_tp_per_cls.get(c, 0)),
                "FN_loc": int(loc_fn_per_cls.get(c, 0)),
                "Recall_loc": (
                    float(loc_tp_per_cls.get(c, 0))
                    / float(loc_tp_per_cls.get(c, 0) + loc_fn_per_cls.get(c, 0))
                    if (loc_tp_per_cls.get(c, 0) + loc_fn_per_cls.get(c, 0)) > 0
                    else 0.0
                ),
                "Precision_loc_pred_class": float(
                    precision_loc_pred_per_cls.get(c, 0.0)
                ),
            }
            for c in labels_map
        ]
    ).set_index("Class")

    # --------------------- Write one CSV ---------------------
    with open(csv_path, "w", newline="") as f:
        f.write("# --- METRICS PER CLASS -------------------------------------------\n")
        df_metrics.to_csv(f, float_format="%.6f")

        f.write(
            "\n# --- CHILD METRICS -----------------------------------------------\n"
        )
        df_child_metrics.to_csv(f, float_format="%.6f")

        f.write(
            "\n# --- CLASS CONFUSION MATRIX RAW ----------------------------------\n"
        )
        df_cm_raw.to_csv(f)

        f.write(
            "\n# --- CLASS CONFUSION MATRIX NORMALIZED ---------------------------\n"
        )
        df_cm_norm.to_csv(f, float_format="%.6f")

        f.write(
            "\n# --- CHILD CONFUSION MATRIX RAW ----------------------------------\n"
        )
        df_child_cm_raw.to_csv(f)

        f.write(
            "\n# --- CHILD CONFUSION MATRIX NORMALIZED ---------------------------\n"
        )
        df_child_cm_norm.to_csv(f, float_format="%.6f")

        f.write(
            "\n# --- LOCALIZATION ONLY (SOTA COMPARABLE) -------------------------------\n"
        )
        f.write(f"loc_TP,{loc_tp}\n")
        f.write(f"loc_FP,{loc_fp}\n")
        f.write(f"loc_FN,{loc_fn}\n")
        f.write(f"loc_Precision,{loc_p:.4f}\n")
        f.write(f"loc_Recall,{loc_r:.4f}\n")
        f.write(f"loc_F1,{loc_f1:.4f}\n")
        f.write(f"fp_in_baby_imgs,{int(results.get('fp_in_baby_imgs', 0))}\n")
        f.write(f"fp_in_adult_imgs,{int(results.get('fp_in_adult_imgs', 0))}\n")
        f.write(f"fp_in_bg_imgs,{int(results.get('fp_in_bg_imgs', 0))}\n")
        f.write("\n")
        df_loc_per_cls.to_csv(f, float_format="%.4f")

    print(f"[INFO] Metrics and confusion matrices saved to {csv_path}")
    return csv_path


def build_image_sizeresolver(dataset, images_subdir: str = "images") -> callable:
    """
    Builds a function that resolves original image dimensions from a dataset.

    Creates a callable that attempts to load and get dimensions of an original image,
    first trying the filename directly and then searching in the dataset's image directory
    with different extensions.

    Args:
        dataset: Dataset object with root_dir and split attributes defining image location
        images_subdir (str, optional): Subdirectory name containing images. Defaults to "images"

    Returns:
        callable: Function that takes a filename (stem or path) and returns:
            - Tuple[int, int]: Original (width, height) if image is found
            - None: If image cannot be found or opened

    Example:
        >>> resolver = build_image_sizeresolver(dataset)
        >>> size = resolver("image001.jpg")  # Returns (1024, 768) or None
    """
    # Construct path to images directory from dataset attributes
    root = Path(dataset.root_dir) / dataset.split / images_subdir

    # Common image file extensions to try if bare filename is provided
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def resolve(fname: str) -> tuple[int, int]:
        p = Path(fname)

        # Strategy 1: Try direct path if it's a complete filepath
        if p.is_file():
            with Image.open(p) as im:
                return im.size  # Returns (width, height)

        # Strategy 2: Try different extensions in dataset image directory
        # Extract stem (filename without extension) to try with different extensions
        stem = p.stem if p.suffix else p.name
        for ext in exts:
            candidate = root / f"{stem}{ext}"
            if candidate.exists():
                with Image.open(candidate) as im:
                    return im.size

        # Return None if image cannot be found or opened
        return None

    return resolve


def load_original_and_scale(dataset, fname: str, resize_size: Tuple[int, int]):
    """
    Load the original image and calculate the scaling factors (sx, sy) relative to the resized dimensions.

    Args:
        dataset: The dataset object containing image metadata and paths.
        fname (str): The filename of the image to load.
        resize_size (Tuple[int, int]): The target size (W_r, H_r) to which the image was resized.

    Returns:
        Tuple[np.ndarray, Tuple[float, float]]: A tuple containing:
            - np_img_rgb (np.ndarray): The original image in RGB format as a NumPy array.
            - (sx, sy) (Tuple[float, float]): Scaling factors for width and height.
              Returns (1.0, 1.0) if the image cannot be loaded.
    """
    # Build a function to resolve the original image dimensions from the dataset
    resolver = build_image_sizeresolver(dataset)

    # Get the original width and height of the image
    wh = resolver(fname)
    if wh is None:
        return None, (
            1.0,
            1.0,
        )  # Return None and default scaling factors if image not found

    W0, H0 = wh  # Original image dimensions (width, height)
    Wr, Hr = resize_size  # Resized image dimensions
    sx = float(W0) / float(Wr)  # Calculate scaling factor for width
    sy = float(H0) / float(Hr)  # Calculate scaling factor for height

    # Load the original image from the dataset's images directory
    root = Path(dataset.root_dir) / dataset.split / "images"
    p = Path(fname)

    # Check if the provided path is a valid file
    if not p.is_file():
        # If not, try to find the image with common extensions
        stem = p.stem if p.suffix else p.name
        for e in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            cand = root / f"{stem}{e}"
            if cand.exists():
                p = cand  # Update path to the found image
                break

    try:
        # Open the image and convert it to RGB format
        with Image.open(p) as im:
            im = im.convert("RGB")
            np_img = np.asarray(im)  # Convert the image to a NumPy array
            return np_img, (sx, sy)  # Return the image and scaling factors
    except Exception:
        return None, (
            1.0,
            1.0,
        )  # Return None and default scaling factors if an error occurs


@torch.inference_mode()
def export_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    anchors_xy: torch.Tensor,
    resize_size: Tuple[int, int],  # (W, H) used by model
    face_thres: float,
    iou_thres: float,
    class_thres: float,
    baby_thres: float,
    device: torch.device,
    labels_map: Dict[int, str],
    out_dir: Path,
    output_scale: str = "original",  # "original" | "resized"
) -> None:
    """
    Export model predictions as annotated images and text files.

    This function performs inference on a dataset and saves:
    1. Annotated images with detected faces and their orientations
    2. Text files with detection coordinates and metadata
    3. Cropped face images per detection

    The output can be saved in two coordinate scales:
    - "original": Native resolution of input images
    - "resized": Model input resolution (e.g., 640x640)

    Args:
        model: The RetinaBabyFace detection model
        loader: DataLoader providing images (and optionally labels)
        anchors_xy: Anchor box coordinates tensor
        resize_size: Model input size as (width, height)
        face_thres: Face detection confidence threshold
        iou_thres: IoU threshold for NMS
        class_thres: Classification confidence threshold
        baby_thres: Baby face confidence threshold
        device: Torch device to run inference on
        labels_map: Dictionary mapping class indices to names
        out_dir: Base output directory path
        output_scale: Whether to save in "original" or "resized" coordinates

    Outputs:
        - out_dir/images/: Directory with annotated images
        - out_dir/labels/: Directory with text files containing detections
        - out_dir/crops/: Directory with cropped face images per class
    """

    # Create output directories
    out_imgs = Path(out_dir) / "images"
    out_lbls = Path(out_dir) / "labels"
    out_crops = Path(out_dir) / "crops"
    for d in (out_imgs, out_lbls, out_crops):
        d.mkdir(parents=True, exist_ok=True)

    # Setup model and anchors
    model.eval()
    anchors_xy = anchors_xy.to(device, non_blocking=True)
    Wr, Hr = resize_size
    nms_image_size = (Hr, Wr)  # NMS expects (H,W) format

    dataset = loader.dataset

    # Statistics counters
    processed = 0  # Total images processed
    saved = 0  # Successfully saved pairs
    empty_batches = 0  # Batches with no valid images
    no_dets = 0  # Images with no detections
    errors = 0  # Failed operations
    crop_attempted = 0
    crop_saved = 0
    crop_failed_none = 0
    crop_failed_exception = 0

    # Print configuration
    tqdm.write(f"Inference on device: {device}")
    tqdm.write(
        f"Dataloader: {len(loader)} batches | batch_size={getattr(loader, 'batch_size', '?')}"
    )
    tqdm.write(f"Output dir: {out_dir}  →  images/, labels/")
    tqdm.write(f"Resize size (W,H): {resize_size} | NMS uses (H,W)={nms_image_size}")
    tqdm.write(f"Output scale: {output_scale}")

    with tqdm(total=len(loader), desc="Batches", unit="batch") as pbar_batches:
        global_idx = 0

        for batch in loader:
            # Move images to device
            imgs = batch["image"].to(device, non_blocking=True)
            if imgs.numel() == 0:
                empty_batches += 1
                pbar_batches.update(1)
                continue

            # Run inference and NMS
            try:
                outputs = infer_with_rotated_nms(
                    model_or_preds=model,
                    images=imgs,
                    anchors_xy=anchors_xy,
                    image_size=nms_image_size,
                    face_thres=face_thres,
                    baby_thres=baby_thres,
                    iou_thres=iou_thres,
                    class_thres=class_thres,
                )
            except Exception as e:
                errors += 1
                tqdm.write(f"Inference error in batch: {e}")
                pbar_batches.update(1)
                continue

            # Process each image in batch
            B = imgs.size(0)
            with tqdm(total=B, desc="Images", leave=False, unit="img") as pbar_imgs:
                for b in range(B):
                    processed += 1

                    # Get robust path for image
                    p = resolve_image_path(batch, b, global_idx, dataset=dataset)
                    stem, ext = p.stem, (p.suffix if p.suffix else ".jpg")

                    # Load base image and compute scale factors
                    try:
                        if output_scale == "original" and p.exists():
                            # Load original image
                            with Image.open(p) as im:
                                im = im.convert("RGB")
                                W0, H0 = im.size
                                base_img = np.asarray(im)
                            sx, sy = float(W0) / float(Wr), float(H0) / float(Hr)
                        else:
                            # Use resized tensor or fallback
                            base_img = denormalize_image(imgs[b])
                            sx, sy = 1.0, 1.0
                            if not ext:
                                ext = ".jpg"
                    except Exception as e:
                        errors += 1
                        tqdm.write(f"Could not prepare base image for {p}: {e}")
                        pbar_imgs.update(1)
                        global_idx += 1
                        continue

                    # Extract predictions
                    try:
                        out_b = outputs[b]

                        boxes_np = to_numpy(out_b.get("boxes"))  # (N,5) -> cx,cy,w,h,theta
                        labels_np = to_numpy(out_b.get("labels"))
                        scores_np = to_numpy(out_b.get("final_score"))
                        polys_np = to_numpy(out_b.get("polygons"))  # (N,8) or (N,4,2)
                        final_keep_np = to_numpy(out_b.get("final_keep"))

                        if final_keep_np is not None and final_keep_np.size > 0:
                            keep = final_keep_np.astype(bool)

                            if boxes_np is not None and boxes_np.size > 0:
                                boxes_np = boxes_np[keep]

                            if labels_np is not None and labels_np.size > 0:
                                labels_np = labels_np[keep]

                            if scores_np is not None and scores_np.size > 0:
                                scores_np = scores_np[keep]

                            if polys_np is not None and polys_np.size > 0:
                                polys_np = polys_np[keep]

                        # Normalize/reconstruct polygons if needed
                        polys_42 = ensure_polygons_42_shape(polys_np)
                        if (
                            (polys_42 is None or polys_42.size == 0)
                            and boxes_np is not None
                            and boxes_np.size > 0
                        ):
                            N = boxes_np.shape[0]
                            polys_42 = np.stack(
                                [xywhr_to_poly42_shape(*boxes_np[i]) for i in range(N)],
                                axis=0,
                                dtype=np.float32,
                            )
                    except Exception as e:
                        errors += 1
                        tqdm.write(f"Postprocess error for {p}: {e}")
                        pbar_imgs.update(1)
                        global_idx += 1
                        continue

                    # Scale coordinates to target output size
                    if output_scale == "original" and (sx != 1.0 or sy != 1.0):
                        polys_for_img = scale_polys(polys_42, sx, sy)
                        boxes_for_txt = (
                            scale_xywhr_boxes(boxes_np, sx, sy)
                            if boxes_np is not None
                            else None
                        )
                        boxes_for_crop = boxes_for_txt
                    else:
                        polys_for_img = polys_42
                        boxes_for_txt = boxes_np
                        boxes_for_crop = boxes_np

                    # Save results
                    try:
                        if polys_for_img is not None and polys_for_img.size > 0:
                            angles = (
                                    boxes_for_crop[:, 4]
                                    if (boxes_for_crop is not None and boxes_for_crop.size > 0)
                                    else np.zeros((0,), dtype=np.float32)
                                )
                            lbls = (
                                labels_np
                                if labels_np is not None
                                else np.zeros((polys_for_img.shape[0],), dtype=np.int64)
                            )
                            scrs = (
                                scores_np
                                if scores_np is not None
                                else np.zeros(
                                    (polys_for_img.shape[0],), dtype=np.float32
                                )
                            )
                            painted = draw_predictions_on_image(
                                base_img=base_img,
                                polygons_xy=polys_for_img,
                                labels=lbls,
                                scores=scrs,
                                angles_rad=angles,
                                labels_map=labels_map,
                            )
                        else:
                            painted = base_img
                            no_dets += 1

                        Image.fromarray(painted).save(out_imgs / f"{stem}{ext}")

                        write_predictions_txt(
                            out_labels_dir=out_lbls,
                            stem=stem,
                            boxes_xywhr=boxes_for_txt,
                            polygons_42=polys_for_img,
                            labels=labels_np,
                            scores=scores_np,
                        )
                        saved += 1

                    except Exception as e:
                        errors += 1
                        tqdm.write(f"Saving error for {p}: {e}")

                    if polys_for_img is not None and polys_for_img.size > 0:
                        for j in range(polys_for_img.shape[0]):
                            crop_attempted += 1

                            try:
                                poly = polys_for_img[j].astype(np.float32)

                                theta = (
                                    float(boxes_for_crop[j, 4])
                                    if (
                                        boxes_for_crop is not None
                                        and boxes_for_crop.size > 0
                                        and j < boxes_for_crop.shape[0]
                                    )
                                    else 0.0
                                )

                                crop_img = get_oriented_face_crop(
                                    base_img=base_img,
                                    poly42=poly,
                                    angle_rad=theta,
                                    desired_scale_crop=1.15,
                                    pivot="tl",
                                    max_output_side=None,
                                )

                                if crop_img is None:
                                    crop_failed_none += 1
                                    tqdm.write(
                                        f"[CROP NONE] {stem} det={j} "
                                        f"theta_deg={math.degrees(theta):.2f} "
                                        f"poly_shape={poly.shape} "
                                        f"xrange=({poly[:,0].min():.1f},{poly[:,0].max():.1f}) "
                                        f"yrange=({poly[:,1].min():.1f},{poly[:,1].max():.1f})"
                                    )
                                    continue

                                cls_idx = (
                                    int(labels_np[j])
                                    if (labels_np is not None and labels_np.size > j)
                                    else 0
                                )
                                cls_name = labels_map.get(cls_idx, str(cls_idx))
                                cls_dir = Path(out_dir) / "crops" / cls_name
                                cls_dir.mkdir(parents=True, exist_ok=True)

                                Image.fromarray(crop_img).save(cls_dir / f"{stem}_{j:02d}.jpg")
                                crop_saved += 1

                            except Exception as e:
                                crop_failed_exception += 1
                                tqdm.write(f"[CROP ERROR] {stem} det={j}: {e}")

                    pbar_imgs.update(1)
                    global_idx += 1

            pbar_batches.update(1)

    # Print summary statistics
    tqdm.write("Export complete")
    tqdm.write(f"   • Processed images : {processed}")
    tqdm.write(f"   • Saved (img+txt)  : {saved}")
    tqdm.write(f"   • No detections    : {no_dets}")
    tqdm.write(f"   • Empty batches    : {empty_batches}")
    tqdm.write(f"   • Errors           : {errors}")
    tqdm.write(f"Images: {out_imgs}")
    tqdm.write(f"Labels: {out_lbls}")
    tqdm.write(f"Crops : {out_crops}")
    tqdm.write(f"   • Crop attempts    : {crop_attempted}")
    tqdm.write(f"   • Crop saved       : {crop_saved}")
    tqdm.write(f"   • Crop None        : {crop_failed_none}")
    tqdm.write(f"   • Crop exceptions  : {crop_failed_exception}")

def export_distance_report(
    distance_rows: List[Dict[str, Any]],
    out_dir: Path,
    fname: str = "distance_report.csv",
) -> Path:
    """
    Export detailed TP/FP/FN matching report with class-distance diagnostics.

    Args:
        distance_rows (List[Dict[str, Any]]): Rows generated in run_evaluation.
        out_dir (Path): Output directory.
        fname (str, optional): CSV filename. Defaults to "distance_report.csv".

    Returns:
        Path: Path to the saved CSV file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / fname

    if len(distance_rows) == 0:
        df_empty = pd.DataFrame(
            columns=[
                "image_id",
                "img_kind",
                "case",
                "gt_idx",
                "pred_idx",
                "gt_class",
                "pred_class",
                "class_distance",
                "opposite_flag",
                "iou",
                "score",
                "is_class_match",
            ]
        )
        df_empty.to_csv(csv_path, index=False)
        print(f"[INFO] Empty distance report saved to {csv_path}")
        return csv_path

    df = pd.DataFrame(distance_rows)

    # Sorting to prioritize potentially problematic rows
    sort_cols = []
    for c in ["case", "opposite_flag", "class_distance", "iou", "score", "image_id"]:
        if c in df.columns:
            sort_cols.append(c)

    if len(sort_cols) > 0:
        ascending = [True] * len(sort_cols)
        # For score and iou, descending is usually more informative
        if "score" in sort_cols:
            ascending[sort_cols.index("score")] = False
        if "iou" in sort_cols:
            ascending[sort_cols.index("iou")] = False
        df = df.sort_values(sort_cols, ascending=ascending)

    df.to_csv(csv_path, index=False)
    print(f"[INFO] Distance report saved to {csv_path}")
    return csv_path


def compute_loc_only_from_final(
    gt_xywhr: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes_final: torch.Tensor,
    pred_scores_final: torch.Tensor,
    pred_labels_final: torch.Tensor,
    iou_threshold: float,
    labels_map: Dict[int, str],
    device: torch.device,
    orient_class_ids_device: torch.Tensor,
) -> Dict[str, Any]:
    """
    Compute localization-only TP/FP/FN with final detections only, ignoring class label
    for matching but preserving per-class counters from GT and predicted labels.

    GT considered for loc-only:
        - Only labels in labels_map (baby orientation classes).
        - Adults (class=-1) are excluded from loc GT denominator.

    Args:
        gt_xywhr: [G_all,5] GT boxes in cx,cy,w,h,theta format for all GT (including non-eval classes)
        gt_labels: [G_all] GT class labels for all GT
        pred_boxes_final: [P,5] Final predicted boxes after NMS in cx,cy,w,h,theta format
        pred_scores_final: [P] Confidence scores for final predicted boxes
        pred_labels_final: [P] Class labels for final predicted boxes
        iou_threshold: IoU threshold to consider a match as TP
        labels_map: Mapping of class ids to names for orientation classes (excludes adults)
        device: Torch device for computations
        orient_class_ids_device: Tensor of class ids that are considered for evaluation (orientation classes)

    Returns:
        Dict with global counts, per-class deltas, and matched/unmatched index lists.
    """
    out = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tp_per_cls": {c: 0 for c in labels_map},
        "fn_per_cls": {c: 0 for c in labels_map},
        "tp_pred_cls": {c: 0 for c in labels_map},
        "fp_pred_cls": {c: 0 for c in labels_map},
        "matched_pairs": [],  # list of tuples (gi_global, pj)
        "unmatched_gt": [],  # gi_global
        "unmatched_pr": [],  # pj
        "n_eval_gt": 0,
    }

    G_all = int(gt_xywhr.size(0))
    P = int(pred_boxes_final.size(0))

    if G_all == 0 and P == 0:
        return out

    if G_all == 0:
        out["fp"] = P
        out["unmatched_pr"] = list(range(P))
        for pj in out["unmatched_pr"]:
            c_pred = int(pred_labels_final[pj].item())
            if c_pred in out["fp_pred_cls"]:
                out["fp_pred_cls"][c_pred] += 1
        return out

    gt_is_eval = (gt_labels.unsqueeze(1) == orient_class_ids_device.unsqueeze(0)).any(
        dim=1
    )
    eval_idx = torch.where(gt_is_eval)[0]
    G = int(eval_idx.numel())
    out["n_eval_gt"] = G

    if G == 0:
        out["fp"] = P
        out["unmatched_pr"] = list(range(P))
        for pj in out["unmatched_pr"]:
            c_pred = int(pred_labels_final[pj].item())
            if c_pred in out["fp_pred_cls"]:
                out["fp_pred_cls"][c_pred] += 1
        return out

    if P == 0:
        out["fn"] = G
        out["unmatched_gt"] = [int(i.item()) for i in eval_idx]
        for gi_global in out["unmatched_gt"]:
            c_gt = int(gt_labels[gi_global].item())
            if c_gt in out["fn_per_cls"]:
                out["fn_per_cls"][c_gt] += 1
        return out

    gt_eval_boxes = gt_xywhr[eval_idx]  # [G,5]
    iou_mat = batch_probiou(gt_eval_boxes, pred_boxes_final)  # [G,P]

    matched_gt_local = torch.zeros(G, dtype=torch.bool, device=device)
    matched_pr = torch.zeros(P, dtype=torch.bool, device=device)

    _, order = torch.sort(pred_scores_final, descending=True)
    for pj in order.tolist():
        if matched_gt_local.all():
            break

        ious = iou_mat[:, pj].clone()
        ious[matched_gt_local] = -1.0
        best_iou, gi_local = ious.max(dim=0)
        best_iou_val = float(best_iou.item())
        gi_local = int(gi_local.item())

        if best_iou_val >= iou_threshold:
            matched_gt_local[gi_local] = True
            matched_pr[pj] = True

    matched_gt_global = eval_idx[matched_gt_local]
    unmatched_gt_global = eval_idx[~matched_gt_local]
    unmatched_pr = torch.where(~matched_pr)[0]

    out["matched_pairs"] = []
    if matched_gt_global.numel() > 0:
        # Rebuild pair list deterministically: nearest matched gt for each matched pred
        # (for detailed report only, not for counters)
        for pj in torch.where(matched_pr)[0].tolist():
            ious = iou_mat[:, pj].clone()
            ious[~matched_gt_local] = -1.0
            _, gi_local = ious.max(dim=0)
            gi_local = int(gi_local.item())
            gi_global = int(eval_idx[gi_local].item())
            out["matched_pairs"].append((gi_global, int(pj)))

    out["unmatched_gt"] = [int(i.item()) for i in unmatched_gt_global]
    out["unmatched_pr"] = [int(i.item()) for i in unmatched_pr]

    tp = int(matched_gt_local.sum().item())
    fn = int((~matched_gt_local).sum().item())
    fp = int((~matched_pr).sum().item())

    out["tp"] = tp
    out["fn"] = fn
    out["fp"] = fp

    for gi_global, pj in out["matched_pairs"]:
        c_gt = int(gt_labels[gi_global].item())
        c_pred = int(pred_labels_final[pj].item())
        if c_gt in out["tp_per_cls"]:
            out["tp_per_cls"][c_gt] += 1
        if c_pred in out["tp_pred_cls"]:
            out["tp_pred_cls"][c_pred] += 1

    for gi_global in out["unmatched_gt"]:
        c_gt = int(gt_labels[gi_global].item())
        if c_gt in out["fn_per_cls"]:
            out["fn_per_cls"][c_gt] += 1

    for pj in out["unmatched_pr"]:
        c_pred = int(pred_labels_final[pj].item())
        if c_pred in out["fp_pred_cls"]:
            out["fp_pred_cls"][c_pred] += 1

    return out
