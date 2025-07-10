import time
import csv
import os
import random
from contextlib import nullcontext
from typing import List, Optional, Dict, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW, RAdam
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import tqdm.auto as tqdm_auto
from torch.nn import functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

tqdm = tqdm_auto.tqdm  # Use tqdm.auto for better compatibility with Jupyter notebooks

from models.anchors import AnchorGeneratorOBB, get_feature_map_shapes
from data_setup.dataset import BabyFacesDataset, calculate_average_obb_dimensions
from data_setup.augmentations import Resize
from loss.utils import xyxyxyxy2xywhr, decode_vertices, batch_probiou
from utils.visualize import denormalize_image
import config


class EarlyStopping:
    """
    EarlyStopping can be used to monitor the validation loss during training and stop the training process early
    if the validation loss does not improve after a certain number of epochs. It can handle both KFold and
    non-KFold cases.
    """

    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0,
        path: str = "checkpoint.pt",
        use_kfold: bool = False,
        trace_func=print,
    ):
        """
        Initializes the EarlyStopping object with the given parameters.

        Args:
            patience: How long to wait after last time validation loss improved.
            verbose: If True, prints a message for each validation loss improvement.
            delta: Minimum change in the monitored quantity to qualify as an improvement.
            path: Path for the checkpoint to be saved to.
            use_kfold: If True, saves the model with the lowest loss metric for each fold.
            trace_func: trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.use_kfold = use_kfold
        self.trace_func = trace_func
        self.fold = None
        self.filename = None

    def __call__(self, val_loss: float, model: nn.Module, fold: int = None):
        """
        This method is called during the training process to monitor the validation loss and decide whether to stop
        the training process early or not.

        Args:
            val_loss: Validation loss of the model at the current epoch.
            model: The PyTorch model being trained.
            fold: The current fold of the KFold cross-validation. Required if use_kfold is True.
        """
        if np.isnan(val_loss):
            self.trace_func("EarlyStopping: val_loss is NaN, skipping...")
            return

        if self.use_kfold:
            assert fold is not None, "Fold must be provided when use_kfold is True"

            # If it's a new fold, resets the early stopping object and sets the filename to save the model
            if fold != self.fold:
                self.fold = fold
                self.counter = 0
                self.best_score = None
                self.early_stop = False
                self.val_loss_min = np.inf
                self.filename = self.path.replace(".pt", f"_fold_{fold}.pt")

        # Calculating the score by negating the validation loss
        score = -val_loss

        # If the best score is None, sets it to the current score and saves the checkpoint
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        # If the score is less than the best score plus delta, increments the counter
        # and checks if the patience has been reached
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

        # If the score is better than the best score plus delta, saves the checkpoint and resets the counter
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module):
        """
        Saves the model when validation loss decreases and it's a numerical value.

        Args:
            val_loss: The current validation loss.
            model: The PyTorch model being trained.
        """
        # If verbose mode is on, print a message about the validation loss decreasing and saving the model
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ..."
            )

        # Save the state of the model to the appropriate filename based on whether KFold is used or not
        if self.use_kfold:
            torch.save(model.state_dict(), self.filename)
        else:
            torch.save(model.state_dict(), self.path)

        # Update the minimum validation loss seen so far to the current validation loss
        self.val_loss_min = val_loss


def nms_rotated(
    boxes: torch.Tensor,  # (N, 5) - (cx, cy, w, h, θ)
    scores: torch.Tensor,  # (N,)   - confidence scores
    threshold: float = 0.45,
    min_area_ratio: float = 0.3,
) -> torch.Tensor:
    """
    Performs rotated Non-Maximum Suppression (NMS) using probabilistic IoU (pIoU) and additional containment/area filtering.

    This function removes redundant rotated bounding boxes based on two criteria:
      1. Suppresses boxes that have high overlap (pIoU ≥ threshold) with a higher-scoring box.
      2. Suppresses boxes that are much smaller (area ratio < min_area_ratio) and are fully contained within a larger box,
         provided their centers are close (distance < 0.2 * sqrt(area_larger)).

    Args:
        boxes (torch.Tensor): Rotated bounding boxes in (cx, cy, w, h, θ) format, shape (N, 5).
        scores (torch.Tensor): Confidence scores for each box, shape (N,).
        threshold (float): IoU threshold for suppression (default: 0.45).
        min_area_ratio (float): Minimum area ratio to consider a box as "small" for containment suppression (default: 0.3).

    Returns:
        torch.Tensor: Indices of boxes to keep after NMS, as a 1D tensor of type long.
    """
    order = scores.argsort(descending=True)
    keep = []

    # While there are boxes to process
    while order.numel() > 0:
        # Select the box with the highest score
        i = order[0].item()
        keep.append(i)

        # If only one box remains, we can stop
        if order.numel() == 1:
            break

        rest = order[1:]
        # Compute pIoU between the current box and the rest
        ious = batch_probiou(boxes[i : i + 1], boxes[rest]).squeeze(0)
        suppress_mask = ious >= threshold

        box_i = boxes[i].unsqueeze(0)  # (1, 5)
        for idx, j in enumerate(rest):
            # Get the box to compare against
            box_j = boxes[j].unsqueeze(0)  # (1, 5)

            # Compute area ratio between boxes
            area_i = box_i[0, 2] * box_i[0, 3]
            area_j = box_j[0, 2] * box_j[0, 3]
            area_ratio = area_j / (area_i + 1e-6)

            # Compute center distance between boxes
            center_dist = torch.norm(box_j[0, :2] - box_i[0, :2])

            # Suppress if box_j is much smaller and close to box_i (likely contained)
            if area_ratio < min_area_ratio and center_dist < 0.2 * torch.sqrt(area_i):
                suppress_mask[idx] = True

        # Keep only boxes not suppressed
        order = rest[~suppress_mask]

    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def infer_with_rotated_nms(
    model_or_preds: Union[nn.Module, Tuple],
    images: torch.Tensor,  # (B, 3, H, W)
    anchors_xy: torch.Tensor,  # (N, 8) anchor corners in xyxyxyxy
    image_size: Tuple[int, int],  # (W, H)
    face_thres: float = 0.20,
    iou_thres: float = 0.45,
    class_thres: float = 0.15,
    alpha_score: float = 0.6,
    pre_nms_topk: int = 1000,
    max_det: int = 300,
) -> List[Dict[str, torch.Tensor]]:
    """
    Performs inference with a RetinaBabyFace-like model that includes an additional face/no-face head,
    applying rotated Non-Maximum Suppression (NMS) to filter predictions.

    Args:
        model_or_preds (Union[nn.Module, Tuple]): Either a model that outputs predictions or a tuple of precomputed outputs.
        images (Tensor): Batch of input images, shape (B, 3, H, W).
        anchors_xy (Tensor): Anchor polygons in (N, 8) format (4 corners per anchor).
        image_size (Tuple[int, int]): Size of input images as (W, H).
        face_thres (float): Minimum face probability to consider a detection.
        iou_thres (float): IoU threshold for rotated NMS.
        class_thres (float): Minimum orientation confidence to consider a detection.
        alpha_score (float): Weighting factor for combining face and orientation confidence.
        pre_nms_topk (int): Maximum number of top-scoring predictions to keep before NMS.
        max_det (int): Maximum number of final predictions per image after NMS.

    Returns:
        List[Dict[str, Tensor]]: List of length B, each dict contains:
            - 'boxes':    (M, 5) boxes in (cx, cy, w, h, θ) format
            - 'scores':   (M,)   combined face/orientation scores
            - 'labels':   (M,)   orientation labels (0–4)
            - 'polygons': (M, 8) 4-corner polygons for visualization
    """
    # If model_or_preds is a model, run inference to get outputs
    if isinstance(model_or_preds, nn.Module):
        orient_logits, face_logits, deltas, pred_angles = model_or_preds(images)
    # If model_or_preds is already a tuple of outputs, unpack them
    else:
        orient_logits, face_logits, deltas, pred_angles = model_or_preds
        # chequeo rápido de forma
        assert orient_logits.shape[0] == images.size(
            0
        ), "Batch size mismatch between model outputs and input images"

    B = images.size(0)

    face_prob = torch.sigmoid(face_logits.squeeze(-1))  # (B, N, 1) → (B, N)
    orientation_probs = F.softmax(orient_logits, dim=-1)  # (B, N, 5)
    outputs = []

    for b in range(B):
        # Get the most probable orientation label and its confidence for each anchor
        orient_conf, orient_labels = orientation_probs[b].max(-1)

        # Compute a combined score using face probability and orientation confidence
        score = (face_prob[b] ** alpha_score) * (orient_conf ** (1 - alpha_score))

        # Filter anchors based on face probability and orientation confidence thresholds
        keep = (face_prob[b] >= face_thres) & (orient_conf >= class_thres)
        if not keep.any():
            outputs.append(
                dict(
                    boxes=torch.empty(0, 5, device=images.device),
                    scores=torch.empty(0, device=images.device),
                    labels=torch.empty(0, device=images.device),
                    polygons=torch.empty(0, 8, device=images.device),
                )
            )
            continue

        # Get indices of anchors that passed the filtering
        idx = keep.nonzero(as_tuple=False).squeeze(1)
        # Select top-K anchors by combined score before NMS
        K = min(pre_nms_topk, idx.numel())
        topk = score[idx].topk(K, sorted=True).indices
        sel = idx[topk]  # (K,)

        # Decode predicted polygons for selected anchors
        verts = decode_vertices(
            deltas[b][sel],
            anchors_xy[sel],
            pred_angles[b][sel].squeeze(-1),
            image_size,
        )  # (K, 8)
        # Convert polygons to (cx, cy, w, h, θ) format
        xywhr = xyxyxyxy2xywhr(
            verts, pred_angles[b][sel].squeeze(-1), image_size
        )  # (K, 5)

        # Apply rotated NMS to filter overlapping predictions
        keep_nms = nms_rotated(xywhr, score[sel], iou_thres)[:max_det]
        sel_final = sel[keep_nms]

        # Prepare output dictionary for this image
        outputs.append(
            {
                "boxes": xywhr[keep_nms],  # (M, 5) in (cx, cy, w, h, θ)
                "scores": score[sel][keep_nms],  # (M,)
                "labels": orient_labels[sel_final].float(),  # (M,)
                "polygons": verts[keep_nms],  # (M, 8) in (x1, y1, ..., x4, y4)
            }
        )

    return outputs


def compute_map_rotated(
    all_pred_boxes: List[torch.Tensor],
    all_pred_scores: List[torch.Tensor],
    all_pred_labels: List[torch.Tensor],
    all_gt_boxes: List[torch.Tensor],
    all_gt_labels: List[torch.Tensor],
    iou_thr: float = 0.5,
    num_classes: int = 5,
) -> float:
    """
    Computes the mean Average Precision (mAP) at a given IoU threshold for rotated bounding boxes (OBBs),
    following the VOC/COCO evaluation protocol. This implementation minimizes Python overhead and performs
    most operations using PyTorch tensors for efficiency.

    Args:
        all_pred_boxes (List[Tensor]): List of tensors [(N_i, 5)] containing predicted boxes for each image.
                                       Each box is in (cx, cy, w, h, θ) format.
        all_pred_scores (List[Tensor]): List of tensors [(N_i,)] containing confidence scores for each prediction.
        all_pred_labels (List[Tensor]): List of tensors [(N_i,)] containing predicted class labels for each box.
        all_gt_boxes (List[Tensor]): List of tensors [(M_i, 5)] containing ground-truth boxes for each image.
                                     Each box is in (cx, cy, w, h, θ) format.
        all_gt_labels (List[Tensor]): List of tensors [(M_i,)] containing ground-truth class labels for each box.
        iou_thr (float): IoU threshold to consider a detection as a "true positive" (default: 0.5).
        num_classes (int): Total number of classes in the dataset (excluding background).

    Returns:
        float: The mean Average Precision (mAP) averaged across all classes.
    """
    # 1) Handle the trivial case: no predictions
    if len(all_pred_scores) == 0:
        return 0.0

    device = all_pred_scores[0].device
    APs: List[float] = []
    eps = 1e-6  # Small epsilon to avoid division by zero

    # 2) Iterate over each class (0 to num_classes-1)
    for c in range(num_classes):
        # 2.1) Collect all predictions for the current class
        boxes_list: List[torch.Tensor] = []
        scores_list: List[torch.Tensor] = []
        imgidx_list: List[torch.Tensor] = []

        for img_i in range(len(all_pred_boxes)):
            mask = all_pred_labels[img_i] == c
            if not mask.any():
                continue
            b_i = all_pred_boxes[img_i][mask]  # (N_i_c, 5)
            s_i = all_pred_scores[img_i][mask]  # (N_i_c,)
            n_i = b_i.shape[0]
            idx_i = torch.full((n_i,), img_i, dtype=torch.long, device=device)
            boxes_list.append(b_i.to(device))
            scores_list.append(s_i.to(device))
            imgidx_list.append(idx_i)

        if len(boxes_list) == 0:
            # If no predictions exist for this class:
            # If there are no ground truths either, AP = 1.0; otherwise, AP = 0.0.
            npos = sum(int((gt_lbl == c).sum().item()) for gt_lbl in all_gt_labels)
            APs.append(1.0 if npos == 0 else 0.0)
            continue

        boxes_c = torch.cat(boxes_list, dim=0)  # (N_tot_c, 5)
        scores_c = torch.cat(scores_list, dim=0)  # (N_tot_c,)
        imgidx_c = torch.cat(imgidx_list, dim=0)  # (N_tot_c,)

        # 2.2) Sort predictions by descending score
        scores_sorted, order = torch.sort(scores_c, descending=True)
        boxes_c = boxes_c[order]
        imgidx_c = imgidx_c[order]
        nD = boxes_c.shape[0]

        # 2.3) Collect all ground truths for the current class, organized by image
        gt_per_img: Dict[int, torch.Tensor] = {}
        detected: Dict[int, torch.Tensor] = {}
        for img_i in range(len(all_gt_boxes)):
            mask_gt = all_gt_labels[img_i] == c
            if mask_gt.any():
                gt_boxes_i = all_gt_boxes[img_i][mask_gt].to(device)  # (M_i_c, 5)
                gt_per_img[img_i] = gt_boxes_i
                detected[img_i] = torch.zeros(
                    gt_boxes_i.shape[0], dtype=torch.bool, device=device
                )
        npos = sum(gt_per_img[i].shape[0] for i in gt_per_img)
        if npos == 0:
            # If no ground truths exist for this class:
            # If there are no predictions either, AP = 1.0; otherwise, AP = 0.0.
            APs.append(1.0 if nD == 0 else 0.0)
            continue

        # 3) Determine True Positives (TP) and False Positives (FP) for each prediction
        tp = torch.zeros(nD, dtype=torch.float32, device=device)
        fp = torch.zeros(nD, dtype=torch.float32, device=device)

        for idx in range(nD):
            img_i = int(imgidx_c[idx].item())
            box_pred = boxes_c[idx : idx + 1]  # (1, 5)

            if img_i not in gt_per_img:
                # No ground truths of this class in the image => pure FP
                fp[idx] = 1.0
                continue

            gt_boxes_img = gt_per_img[img_i]  # (M_i_c, 5)
            ious = batch_probiou(box_pred, gt_boxes_img)  # (1, M_i_c)
            best_iou, best_j = ious[0].max(dim=0)

            if (best_iou.item() >= iou_thr) and (not detected[img_i][best_j]):
                # Valid match & unmatched GT => TP
                tp[idx] = 1.0
                detected[img_i][best_j] = True
            else:
                # IoU < threshold or GT already matched => FP
                fp[idx] = 1.0

        # 4) If no valid predictions exist, AP = 0
        if nD == 0:
            APs.append(0.0)
            continue

        # 5) Build Precision-Recall curve and compute AP
        tp_cum = torch.cumsum(tp, dim=0)  # Cumulative TP (N_tot_c,)
        fp_cum = torch.cumsum(fp, dim=0)  # Cumulative FP (N_tot_c,)

        recall = tp_cum / float(npos + eps)  # Recall (N_tot_c,)
        precision = tp_cum / (tp_cum + fp_cum + eps)  # Precision (N_tot_c,)

        # Add initial (recall=0, precision=1) and final (recall=1, precision=0) points
        recall = torch.cat(
            [torch.zeros(1, device=device), recall, torch.ones(1, device=device)]
        )
        precision = torch.cat(
            [torch.ones(1, device=device), precision, torch.zeros(1, device=device)]
        )

        # Ensure precision is non-decreasing
        for i in range(precision.shape[0] - 2, -1, -1):
            if precision[i] < precision[i + 1]:
                precision[i] = precision[i + 1]

        # Compute AP using the trapezoidal rule
        delta_rec = recall[1:] - recall[:-1]
        ap_c = torch.sum(delta_rec * precision[1:]).item()
        APs.append(ap_c)

    # 6) Compute mAP as the mean of all APs
    if len(APs) == 0:
        return 0.0
    return float(np.mean(APs))


def get_resize_size(dataloader: DataLoader) -> Tuple[int, int]:
    """
    Gets the resize size from the dataloader by iterating over the first batch.
    Assumes that the dataloader returns a dictionary with an "image" key.
    Args:
        dataloader (DataLoader): The dataloader to get the resize size from.
    Returns:
        Tuple[int, int]: The resize size (width, height).
    """
    # Get the first batch from the dataloader
    sample = next(iter(dataloader))
    # Assuming the dataloader returns a dictionary with an "image" key
    # and the image is in the format (C, H, W)
    # Get the image shape
    H, W = sample["image"].shape[-2:]
    return (W, H)


def get_base_obb_stats(
    resize_size: Tuple[int, int],
    obb_stats_by_size: Dict[Tuple[int, int], Dict[str, float]],
    root_dir: Optional[str] = None,
) -> Tuple[float, float]:
    """Retrieves base size and ratio for OBB generation, using precomputed stats or calculating from dataset.

    Automatically uses precomputed statistics when available for the given image size.
    Falls back to dataset calculation only when necessary and when root_dir is provided.

    Args:
        resize_size: Target image dimensions (width, height)
        obb_stats_by_size: Dictionary mapping image sizes to precomputed OBB statistics
        root_dir: Optional dataset path (required only if precomputed stats unavailable)

    Returns:
        Tuple containing (base_size, base_ratio)

    Raises:
        ValueError: If precomputed stats unavailable and root_dir not provided
    """
    # First try to use precomputed statistics
    if resize_size in obb_stats_by_size:
        stats = obb_stats_by_size[resize_size]
        print(
            f"[INFO] Using precomputed OBB stats for size {resize_size}: "
            f"size={stats['avg_size']:.2f}, ratio={stats['avg_ratio']:.2f}"
        )
        return stats["avg_size"], stats["avg_ratio"]

    # Fall back to dataset calculation if root_dir provided
    if root_dir is not None:
        print(f"[INFO] Computing OBB statistics from dataset (resize={resize_size})...")
        raw_dataset = BabyFacesDataset(
            root_dir=root_dir, split="train", transform=Resize(resize_size)
        )

        stats = calculate_average_obb_dimensions(raw_dataset, resize_size)
        print(
            f"[INFO] Computed OBB stats: size={stats['avg_size']:.2f}, "
            f"ratio={stats['avg_ratio']:.2f}"
        )
        return stats["avg_size"], stats["avg_ratio"]

    raise ValueError(
        f"No precomputed stats available for size {resize_size} and "
        "root_dir not provided for dataset calculation"
    )


def generate_anchors_for_training(
    model: nn.Module,
    resize_size: Tuple[int, int],
    device: torch.device,
    scale_factors: List[float],
    ratio_factors: List[float],
    anchor_preview_path: Optional[Union[str, Path]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates oriented anchor boxes (OBBs) for the training stage of the model, based on the
    feature map resolutions and the anchor generation configuration.

    This function infers the feature map sizes from the model's architecture given a specified input
    resolution. It then uses an OBB anchor generator to create rotated anchor boxes in both
    vertex-based (xyxyxyxy) and parameterized (cx, cy, w, h, angle) formats. A visual preview of the anchors
    is optionally saved as an image.

    Args:
        model (nn.Module): The model from which to extract the feature map shapes.
        resize_size (Tuple[int, int]): The target input image size as (width, height).
        device (torch.device): Device on which tensors are created (CPU or GPU).
        scale_factors (List[float]): List of scale multipliers to generate anchors of various sizes.
        ratio_factors (List[float]): List of aspect ratio multipliers to create anchors of different shapes.
        anchor_preview_path (Optional[Union[str, Path]]): Path to save a preview of the generated anchors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - anchors_xy (torch.Tensor): Anchor boxes in vertex format (N, 8).
            - anchors_xywhr (torch.Tensor): Anchor boxes in (cx, cy, w, h, angle) format (N, 5).
    """
    # Get the output feature map shapes for each FPN level from the model
    feature_shapes = get_feature_map_shapes(
        model, input_shape=(1, 3, resize_size[1], resize_size[0])
    )

    # Compute the stride (downsampling factor) for each feature map
    strides = [int(round(resize_size[1] / h)) for (h, w) in feature_shapes]

    # Initialize the oriented bounding box anchor generator
    anchors_per_level = []
    for lvl_idx, ((h, w), stride) in enumerate(zip(feature_shapes, strides)):
        base = config.BASE_ANCHOR_SIZES[
            lvl_idx
        ]  # Base anchor size for the current level

        # Create an anchor generator for the current level
        level_gen = AnchorGeneratorOBB(
            base_size=base,
            base_ratio=1.0,
            scale_factors=scale_factors,  # Scale multipliers for anchor sizes
            ratio_factors=ratio_factors,  # Aspect ratio multipliers for anchor shapes
            angles=config.ANGLES,  # List of rotation angles for anchors
        )

        # Generate anchors for the current feature map level
        anc_i = level_gen.generate_anchors(
            feature_map_shapes=[(h, w)],
            strides=[stride],
            device=device,
        )  # → Tensor (h * w * num_anchors_per_cell, 8)

        anchors_per_level.append(anc_i)

    # Concatenate anchors from all feature map levels
    anchors_xy = torch.cat(
        anchors_per_level, dim=0
    )  # (∑ h_i*w_i*num_anchors_per_cell, 8)

    # Convert the anchors to parameterized (cx, cy, w, h, angle) format
    zeros = torch.zeros(len(anchors_xy), device=device)  # No angle during generation
    anchors_xywhr = xyxyxyxy2xywhr(anchors_xy, zeros, (resize_size[0], resize_size[1]))

    # Optionally save a preview of a sample of anchors
    if anchor_preview_path is not None and not os.path.exists(anchor_preview_path):
        all_anc = anchors_xy.cpu().numpy()  # (N, 8)
        K = min(500, all_anc.shape[0])  # Sample K anchors for visualization
        idxs = random.sample(range(all_anc.shape[0]), K)

        # Use HSV colormap for diverse colors
        cmap = plt.cm.get_cmap("hsv", K)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_xlim(0, resize_size[0])
        ax.set_ylim(resize_size[1], 0)
        ax.set_title("Anchor preview")
        ax.axis("off")

        # Draw each anchor as a colored polygon
        for j, i in enumerate(idxs):
            pts = all_anc[i].reshape(4, 2)
            color = cmap(j)
            poly = MplPolygon(
                pts, closed=True, fill=False, edgecolor=color, linewidth=0.8
            )
            ax.add_patch(poly)

        plt.tight_layout()
        plt.savefig(anchor_preview_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Anchor preview saved to {anchor_preview_path}")

    return anchors_xy, anchors_xywhr


def create_optimizer(
    which_optimizer: str, model: nn.Module, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    """
    Creates and returns an optimizer for the model.

    Args:
        which_optimizer (str): The optimizer to use ('ADAM', 'SGD', 'ADAMW', or 'RAdam').
        model (nn.Module): The model whose parameters will be optimized.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay (L2 regularization).

    Returns:
        torch.optim.Optimizer: Instantiated optimizer.

    Raises:
        ValueError: If the optimizer name is not recognized.
    """
    # Select optimizer based on string identifier
    if which_optimizer == "ADAM":
        return Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            amsgrad=True,
        )
    elif which_optimizer == "SGD":
        return SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    elif which_optimizer == "ADAMW":
        return AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            amsgrad=True,
        )
    elif which_optimizer == "RAdam":
        return RAdam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(
            "The optimizer must be one of: 'ADAM', 'SGD', 'ADAMW', or 'RAdam'"
        )


def create_scheduler(
    which_scheduler: Optional[str],
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    epochs: int,
    train_dataloader: DataLoader,
) -> Optional[lr_scheduler._LRScheduler]:
    """
    Creates and returns a learning rate scheduler for the optimizer.

    Args:
        which_scheduler (Optional[str]): Scheduler type: 'ReduceLR', 'OneCycle', 'Cosine', or None.
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate will be scheduled.
        learning_rate (float): Initial learning rate.
        epochs (int): Total number of training epochs.
        train_dataloader (DataLoader): Training dataloader (used for steps per epoch).

    Returns:
        Optional[lr_scheduler._LRScheduler]: Instantiated scheduler, or None if not used.

    Raises:
        ValueError: If the scheduler type is not recognized.
    """
    if which_scheduler == "ReduceLR":
        # Reduce learning rate when a metric has stopped improving
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=3, min_lr=1e-5
        )
    elif which_scheduler == "OneCycle":
        # OneCycleLR: increases LR up to max_lr, then decreases it, for superconvergence
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 3,
            epochs=epochs,
            steps_per_epoch=len(train_dataloader),
        )
    elif which_scheduler == "Cosine":
        # CosineAnnealingLR with linear warmup for the first few epochs
        warmup_epochs = 3
        total_iters_warmup = warmup_epochs * len(train_dataloader)
        total_iters_cosine = (epochs - warmup_epochs) * len(train_dataloader)

        warmup = lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,  # Start at 10% of initial LR
            end_factor=1.0,
            total_iters=total_iters_warmup,
        )

        cosine = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_iters_cosine,
            eta_min=1e-6,
        )

        # SequentialLR: first warmup, then cosine annealing
        return lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[total_iters_warmup],
        )
    elif which_scheduler is None:
        # No scheduler used
        return None
    else:
        raise ValueError(
            "The scheduler for learning rate must be either 'ReduceLR', 'OneCycle', or 'Cosine'"
        )


def build_multitask_targets(
    batch_targets: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Prepares and formats the target dictionary for the multitask model by moving tensors to the desired device
    and adjusting shapes as needed.

    Args:
        batch_targets (Dict[str, torch.Tensor]): Dictionary containing:
            - "boxes"      : (B, N, 8)  -> Oriented bounding boxes (vertices).
            - "angles"     : (B, N)     -> Rotation angles in radians.
            - "class_idx"  : (B, N)     -> Class labels per box.
            - "valid_mask" : (B, N)     -> Mask indicating valid boxes.
        device (torch.device): Device to move all tensors to (e.g., CUDA or CPU).

    Returns:
        Dict[str, torch.Tensor]: Dictionary with preprocessed targets for loss computation:
            - "class_idx"  : (B, N)
            - "boxes"      : (B, N, 8)
            - "angle"      : (B, N, 1)
            - "valid_mask" : (B, N)
    """
    return {
        "class_idx": batch_targets["class_idx"].to(device),  # (B, N)
        "boxes": batch_targets["boxes"].to(device),  # (B, N, 8)
        "angle": batch_targets["angles"].unsqueeze(-1).to(device),  # (B, N, 1)
        "valid_mask": batch_targets["valid_mask"].to(device),  # (B, N)
    }


def train_step(
    model: nn.Module,
    train_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    clip_value: float,
    grad_clip_mode: str,
    scheduler: lr_scheduler._LRScheduler,
    device: torch.device,
    anchors: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[float, float, float, float, float]:
    """
    Performs a single training step for the model.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        loss_fn (nn.Module): Loss function for the model.
        optimizer (Optimizer): Optimizer for the model.
        clip_value (float): Value for gradient clipping.
        grad_clip_mode (str): Mode for gradient clipping ("Norm" or "Value").
        scheduler (lr_scheduler._LRScheduler): Learning rate scheduler.
        device (torch.device): Device to use for training.
        anchors (torch.Tensor): Anchor boxes tensor.

    Returns:
        Tuple[float, float, float, float, float, float]:
        - Average total loss
        - Average class loss
        - Average face loss
        - Average OBB loss
        - Average angular loss
        - And current learning rate.
    """
    model.train()  # Set the model to training mode.
    total_loss_sum = 0.0
    face_loss_sum = 0.0
    class_loss_sum = 0.0
    obb_loss_sum = 0.0
    angular_loss_sum = 0.0
    rect_loss_sum = 0.0
    total_batches = 0

    # Enable Automatic Mixed Precision (AMP) if running on CUDA for faster training
    use_amp = device.type == "cuda"

    if use_amp:
        try:
            from torch.amp import GradScaler, autocast  # PyTorch >= 2.0
        except ImportError:
            from torch.cuda.amp import GradScaler, autocast  # PyTorch < 2.0 fallback
        scaler = GradScaler()  # Scales gradients to prevent underflow in float16
        autocast_context = autocast(
            device_type="cuda", enabled=True
        )  # Context manager for mixed precision
        print("[INFO] Using Automatic Mixed Precision (AMP) for training.")
    else:
        scaler = None
        autocast_context = nullcontext()  # No-op context for CPU or no AMP

    # Progress bar for training batches
    bar = tqdm(
        train_dataloader, desc="  Train", unit="batch", leave=False, dynamic_ncols=True
    )

    for batch in bar:
        images = batch["image"].to(device)  # Move images to device (CPU/GPU)
        targets_raw = batch["target"]
        targets = build_multitask_targets(
            targets_raw, device
        )  # Prepare targets for loss

        optimizer.zero_grad()  # Reset gradients before backward pass
        anchors_xy, anchors_xywhr = anchors
        batch_anchors = anchors_xy.unsqueeze(0).repeat(
            images.size(0), 1, 1
        )  # Expand anchors for batch
        image_sizes = [(images.shape[3], images.shape[2])] * images.size(
            0
        )  # List of image sizes per batch

        with autocast_context:  # Enable mixed precision if AMP is used
            pred = model(images)  # Forward pass
            loss, loss_class, loss_face, loss_obb, loss_angle, loss_rect = loss_fn(
                pred, targets, batch_anchors, anchors_xywhr, image_sizes
            )

        if use_amp:
            scaler.scale(loss).backward()  # Backward pass with gradient scaling

            if clip_value is not None:
                scaler.unscale_(optimizer)  # Unscale gradients before clipping
                if grad_clip_mode == "Norm":
                    clip_grad_norm_(model.parameters(), clip_value)  # Clip by norm
                elif grad_clip_mode == "Value":
                    clip_grad_value_(model.parameters(), clip_value)  # Clip by value

            scaler.step(optimizer)  # Optimizer step with scaled gradients
            scaler.update()  # Update the scaler for next iteration
        else:
            loss.backward()  # Standard backward pass

            if clip_value is not None:
                if grad_clip_mode == "Norm":
                    clip_grad_norm_(model.parameters(), clip_value)  # Clip by norm
                elif grad_clip_mode == "Value":
                    clip_grad_value_(model.parameters(), clip_value)  # Clip by value

            optimizer.step()  # Optimizer step

        # Step the scheduler if it's not ReduceLROnPlateau
        # ReduceLROnPlateau requires a separate step with validation loss
        # scheduler.step(loss) is called in the validation step.
        # For other schedulers, we step it here after optimizer step.
        if scheduler is not None and not isinstance(
            scheduler, lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step()

        # Accumulate losses for reporting
        total_loss_sum += loss.item()
        class_loss_sum += loss_class
        face_loss_sum += loss_face
        obb_loss_sum += loss_obb
        angular_loss_sum += loss_angle
        rect_loss_sum += loss_rect
        total_batches += 1

    bar.close()

    # Compute average losses and current learning rate for reporting
    current_lr = optimizer.param_groups[0]["lr"]  # Get current learning rate.
    avg_total_loss = total_loss_sum / total_batches
    avg_class_loss = class_loss_sum / total_batches
    avg_face_loss = face_loss_sum / total_batches
    avg_obb_loss = obb_loss_sum / total_batches
    avg_angular_loss = angular_loss_sum / total_batches
    avg_rect_loss = rect_loss_sum / total_batches

    return (
        avg_total_loss,
        avg_class_loss,
        avg_face_loss,
        avg_obb_loss,
        avg_angular_loss,
        avg_rect_loss,
        current_lr,
    )


def val_step(
    model: nn.Module,
    val_dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    anchors: Tuple[torch.Tensor, torch.Tensor],
    face_thres: float = 0.25,
    iou_thres: float = 0.5,
    class_thres: float = 0.6,
    alpha_score: float = 0.7,
) -> Tuple[float, float, float, float, float, float]:
    """
    Runs one full evaluation loop on the validation dataset, computing predictions, losses, and rotated mAP.

    Args:
        model (nn.Module): The trained model to evaluate.
        val_dataloader (DataLoader): DataLoader that provides batches of validation data.
        loss_fn (nn.Module): Multi-task loss function that returns total and sub-losses.
        device (torch.device): The device (CPU/GPU) on which computation will be performed.
        anchors (Tuple[Tensor, Tensor]): A tuple containing:
            - anchors_xy (Tensor): Tensor of base anchor vertices (N, 8).
            - anchors_xywhr (Tensor): Tensor of anchors in (cx, cy, w, h, θ) format (N, 5).
        face_thres (float): Confidence threshold for face detection.
        iou_thres (float): IoU threshold for rotated NMS.
        class_thres (float): Confidence threshold for class predictions
        alpha_score (float): Weighting factor for combining face and class scores.

    Returns:
        Tuple[float, float, float, float, float, float]: A tuple containing:
            - avg_loss (float): Average total loss across all validation batches.
            - avg_class_loss (float): Average classification loss.
            - avg_face_loss (float): Average face loss.
            - avg_obb_loss (float): Average oriented bounding box (OBB) loss.
            - avg_angular_loss (float): Average angular prediction loss.
            - mAP (float): Mean Average Precision (mAP) for rotated bounding boxes.
    """
    model.eval()  # Switch model to evaluation mode (no dropout, batchnorm is fixed).

    # Initialize accumulators for different losses
    total_loss = 0.0
    class_loss_sum = 0.0
    face_loss_sum = 0.0
    obb_loss_sum = 0.0
    angular_loss_sum = 0.0
    rect_loss_sum = 0.0
    total_batches = 0

    # Prepare containers to collect predictions and ground truths
    all_pred_boxes, all_pred_scores, all_pred_labels = [], [], []
    all_gt_boxes, all_gt_labels = [], []

    # Disable gradient computation for faster inference and lower memory usage
    with torch.inference_mode():
        bar = tqdm(
            val_dataloader, desc="   Val", unit="batch", leave=False, dynamic_ncols=True
        )
        for batch in bar:
            images = batch["image"].to(device)  # Move images to the target device
            targets_raw = batch["target"]
            targets = build_multitask_targets(targets_raw, device)  # Prepare targets

            anchors_xy, anchors_xywhr = anchors
            batch_anchors = anchors_xy.unsqueeze(0).repeat(images.size(0), 1, 1)

            preds = model(images)

            # Perform inference and apply rotated NMS
            outputs = infer_with_rotated_nms(
                preds,
                images,
                anchors_xy,
                image_size=(images.shape[3], images.shape[2]),
                face_thres=face_thres,
                iou_thres=iou_thres,
                class_thres=class_thres,
                alpha_score=alpha_score,
            )

            # Accumulate predictions and ground truths for each image in the batch
            for b, out in enumerate(outputs):
                # Predictions retained by NMS in (cx, cy, w, h, θ) format
                all_pred_boxes.append(out["boxes"].cpu().detach())
                all_pred_scores.append(out["scores"].cpu().detach())
                all_pred_labels.append(out["labels"].cpu().detach().long())

                # Ground truth: filter by valid_mask and convert to (cx, cy, w, h, θ)
                keep = targets["valid_mask"][b]
                if keep.any():
                    gt_polygons = targets["boxes"][b][keep]  # (M_gt, 8)
                    gt_angles = targets["angle"][b][keep].squeeze(-1)  # (M_gt,)
                    gt_labels = targets["class_idx"][b][keep]  # (M_gt,)

                    gt_xywhr = xyxyxyxy2xywhr(
                        gt_polygons,
                        gt_angles.unsqueeze(-1),
                        (images.shape[3], images.shape[2]),
                    )
                    all_gt_boxes.append(gt_xywhr.cpu().detach())
                    all_gt_labels.append(gt_labels.cpu().detach().long())
                else:
                    # If no ground truth exists for this image, pass an empty tensor
                    all_gt_boxes.append(torch.zeros((0, 5), dtype=torch.float32))
                    all_gt_labels.append(torch.zeros((0,), dtype=torch.long))

            # Compute loss for the batch
            image_sizes = [(images.shape[3], images.shape[2])] * images.size(0)
            loss, loss_class, loss_face, loss_obb, loss_angle, loss_rect = loss_fn(
                preds, targets, batch_anchors, anchors_xywhr, image_sizes
            )
            total_loss += loss.item()
            class_loss_sum += loss_class
            face_loss_sum += loss_face
            obb_loss_sum += loss_obb
            angular_loss_sum += loss_angle
            rect_loss_sum += loss_rect
            total_batches += 1

    bar.close()

    # Compute average losses
    avg_loss = total_loss / total_batches
    avg_class_loss = class_loss_sum / total_batches
    avg_face_loss = face_loss_sum / total_batches
    avg_obb_loss = obb_loss_sum / total_batches
    avg_ang_loss = angular_loss_sum / total_batches
    avg_rect_loss = rect_loss_sum / total_batches

    # Compute rotated mAP using accumulated predictions and ground truths
    mAP = compute_map_rotated(
        all_pred_boxes,
        all_pred_scores,
        all_pred_labels,
        all_gt_boxes,
        all_gt_labels,
        iou_thr=iou_thres,
        num_classes=5,
    )

    return (
        avg_loss,
        avg_class_loss,
        avg_face_loss,
        avg_obb_loss,
        avg_ang_loss,
        avg_rect_loss,
        mAP,
    )


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss_fn: nn.Module,
    which_optimizer: str,
    weight_decay: float,
    learning_rate: float,
    epochs: int,
    device: torch.device,
    early_stopping=None,
    which_scheduler: str = None,
    clip_value: float = None,
    grad_clip_mode: str = None,
    record_metrics: bool = False,
    project: str = "My_WandB_Project",
    run_name: str = "My_Run",
    scale_factors: List[float] = [0.5, 0.75, 1.0, 1.5],
    ratio_factors: List[float] = [0.85, 1.0, 1.15],
    face_thres: float = 0.25,
    iou_thres: float = 0.5,
    class_thres: float = 0.6,
    alpha_score: float = 0.7,
    grid_shape: Tuple[int, int] = (3, 3),
    csv_path: Union[str, Path] = "training_metrics.csv",
    anchor_preview_path: Optional[Union[str, Path]] = None,
    inference_preview: Optional[Union[str, Path]] = None,
    show_every_epoch: int = 5,
) -> Dict[str, List[float]]:
    """
    Trains the model and optionally records metrics.

    This function handles the training loop, validation, and logging of metrics. It supports early stopping,
    learning rate scheduling, gradient clipping, and anchor generation for training. Additionally, it can
    save training metrics to a CSV file, generate anchor previews, and perform qualitative inference during
    training. Optionally, it integrates with Weights & Biases for tracking metrics and visualizations.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        loss_fn (nn.Module): Loss function for the model.
        which_optimizer (str): Optimizer to use ('ADAM' or 'SGD').
        weight_decay (float): Weight decay for the optimizer.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        device (torch.device): Device to use for training.
        early_stopping: Early stopping object (optional).
        which_scheduler (str, optional): Learning rate scheduler to use ('ReduceLR', 'OneCycle', 'Cosine', or None).
        clip_value (float, optional): Value for gradient clipping.
        grad_clip_mode (str, optional): Mode for gradient clipping ('Norm' or 'Value').
        record_metrics (bool, optional): Whether to record metrics using Weights & Biases.
        project (str, optional): Weights & Biases project name.
        run_name (str, optional): Weights & Biases run name.
        scale_factors (List[float], optional): Scale factors for anchor generation.
        ratio_factors (List[float], optional): Ratio factors for anchor generation.
        face_thres (float, optional): Face detection confidence threshold for filtering predictions.
        iou_thres (float, optional): IoU threshold for rotated NMS.
        class_thres (float, optional): Class confidence threshold for filtering predictions.
        grid_shape (Tuple[int, int], optional): Grid shape for inference visualization (rows, cols).
        csv_path (Union[str, Path], optional): Path to save training metrics CSV.
        alpha_score (float): Weighting factor for combining face and orientation confidence.
        anchor_preview_path (Union[str, Path], optional): Path to save anchor preview image.
        inference_preview (Union[str, Path], optional): Path to save inference preview image.
        show_every_epoch (int, optional): Frequency of showing inference previews during training.

    Returns:
        Dict[str, List[float]]: Dictionary containing lists of training and validation metrics.
    """

    # Prepare CSV file for logging metrics
    csv_filename = str(csv_path)
    header = [
        "epoch",
        "train_total_loss",
        "train_class_loss",
        "train_face_loss",
        "train_obb_loss",
        "train_angular_loss",
        "train_rect_loss",
        "test_total_loss",
        "test_class_loss",
        "test_face_loss",
        "test_obb_loss",
        "test_angular_loss",
        "test_rect_loss",
        "test_mAP",
        "learning_rate",
        "epoch_time",
    ]
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # Initialize results dictionary to store metrics
    results = {
        "train_total_loss": [],
        "train_class_loss": [],
        "train_face_loss": [],
        "train_obb_loss": [],
        "train_angular_loss": [],
        "train_rect_loss": [],
        "test_total_loss": [],
        "test_class_loss": [],
        "test_face_loss": [],
        "test_obb_loss": [],
        "test_angular_loss": [],
        "test_rect_loss": [],
        "test_mAP": [],
    }

    model.to(device)  # Move model to the specified device.
    optimizer = create_optimizer(
        which_optimizer=which_optimizer,
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )  # Create optimizer.

    scheduler = create_scheduler(
        which_scheduler=which_scheduler,
        optimizer=optimizer,
        learning_rate=learning_rate,
        epochs=epochs,
        train_dataloader=train_dataloader,
    )  # Create learning rate scheduler.

    if grad_clip_mode:
        assert grad_clip_mode in [
            "Norm",
            "Value",
        ], "grad_clip_mode must be 'Norm' or 'Value'"  # Validate gradient clipping mode.

    # Get the resize size from the dataloader
    resize_size = get_resize_size(train_dataloader)

    # Generate anchors for training
    anchors_xy, anchors_xywhr = generate_anchors_for_training(
        model=model,
        resize_size=resize_size,
        device=device,
        scale_factors=scale_factors,
        ratio_factors=ratio_factors,
        anchor_preview_path=anchor_preview_path,
    )
    anchors_tuple = (anchors_xy, anchors_xywhr)

    start_time = time.time()
    if record_metrics:
        wandb.init(project=project, name=run_name)  # Initialize Weights & Biases.
        wandb.watch(model, loss_fn, log="all")  # Watch model and loss function.

    try:
        for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
            epoch_start = time.time()

            # Perform a training step
            (
                train_total_loss,
                train_class_loss,
                train_face_loss,
                train_obb_loss,
                train_angular_loss,
                train_rect_loss,
                current_lr,
            ) = train_step(
                model=model,
                train_dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                clip_value=clip_value,
                grad_clip_mode=grad_clip_mode,
                scheduler=scheduler,
                device=device,
                anchors=anchors_tuple,
            )

            # Perform a validation step
            (
                test_total_loss,
                test_class_loss,
                test_face_loss,
                test_obb_loss,
                test_angular_loss,
                test_rect_loss,
                test_mAP,
            ) = val_step(
                model=model,
                val_dataloader=val_dataloader,
                loss_fn=loss_fn,
                device=device,
                anchors=anchors_tuple,
                face_thres=face_thres,
                iou_thres=iou_thres,
                class_thres=class_thres,
                alpha_score=alpha_score,
            )

            # Update scheduler if applicable
            if scheduler is not None:
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_total_loss)
                # elif isinstance(scheduler, lr_scheduler.CosineAnnealingLR):
                #     scheduler.step() # Cosine scheduler does not need step here
                # since T_max is set to total iterations

            epoch_time = time.time() - epoch_start
            print(
                f"\nEpoch {epoch+1} | LR: {current_lr:.6f} | Time: {epoch_time//60:.0f}m {epoch_time%60:.2f}s"
            )
            print(
                f"Train metrics | Train Loss: {train_total_loss:.4f} | Class Loss: {train_class_loss:.4f} | Face Loss: {train_face_loss:.4f} | OBB Loss: {train_obb_loss:.4f} | Angle Loss: {train_angular_loss:.4f} | Rect Loss: {train_rect_loss:.4f}"
            )
            print(
                f"Test metrics | Test Loss: {test_total_loss:.4f} | Class Loss: {test_class_loss:.4f} | Face Loss: {test_face_loss:.4f} | OBB Loss: {test_obb_loss:.4f} | Angle Loss: {test_angular_loss:.4f} | Rect Loss: {test_rect_loss:.4f} | mAP: {test_mAP:.4f}"
            )

            if record_metrics:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_total_loss": train_total_loss,
                        "train_class_loss": train_class_loss,
                        "train_face_loss": train_face_loss,
                        "train_obb_loss": train_obb_loss,
                        "train_angular_loss": train_angular_loss,
                        "train_rect_loss": train_rect_loss,
                        "test_total_loss": test_total_loss,
                        "test_class_loss": test_class_loss,
                        "test_face_loss": test_face_loss,
                        "test_obb_loss": test_obb_loss,
                        "test_angular_loss": test_angular_loss,
                        "test_rect_loss": test_rect_loss,
                        "test_mAP": test_mAP,
                        "learning_rate": current_lr,
                        "epoch_time": epoch_time,
                    }
                )  # Log metrics to Weights & Biases.

            # Update results dictionary
            results["train_total_loss"].append(train_total_loss)
            results["train_class_loss"].append(train_class_loss)
            results["train_face_loss"].append(train_face_loss)
            results["train_obb_loss"].append(train_obb_loss)
            results["train_angular_loss"].append(train_angular_loss)
            results["train_rect_loss"].append(train_rect_loss)
            results["test_total_loss"].append(test_total_loss)
            results["test_class_loss"].append(test_class_loss)
            results["test_face_loss"].append(test_face_loss)
            results["test_obb_loss"].append(test_obb_loss)
            results["test_angular_loss"].append(test_angular_loss)
            results["test_rect_loss"].append(test_rect_loss)
            results["test_mAP"].append(test_mAP)

            # Write metrics to CSV file
            with open(csv_filename, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch + 1,
                        f"{train_total_loss:.4f}",
                        f"{train_class_loss:.4f}",
                        f"{train_face_loss:.4f}",
                        f"{train_obb_loss:.4f}",
                        f"{train_angular_loss:.4f}",
                        f"{train_rect_loss:.4f}",
                        f"{test_total_loss:.4f}",
                        f"{test_class_loss:.4f}",
                        f"{test_face_loss:.4f}",
                        f"{test_obb_loss:.4f}",
                        f"{test_angular_loss:.4f}",
                        f"{test_rect_loss:.4f}",
                        f"{test_mAP:.4f}",
                        f"{current_lr:.5f}",
                        f"{epoch_time:.4f}",
                    ]
                )

            # Save inference preview every few epochs
            if (epoch + 1) % show_every_epoch == 0 and inference_preview is not None:
                out_path = inference_preview / f"{run_name}_epoch{epoch+1}.jpg"
                in_training_inference(
                    model,
                    val_dataloader,
                    anchors_xy,
                    device,
                    resize_size,
                    out_path,
                    grid_shape,
                )

            # Check early stopping condition
            if early_stopping is not None:
                early_stopping(test_total_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
    finally:
        if record_metrics:
            wandb.finish()  # Finish Weights & Biases run.

    elapsed_time = time.time() - start_time
    print(
        f"[INFO] Total training time: {elapsed_time//60:.0f} minutes, {elapsed_time%60:.2f} seconds"
    )
    return results


def in_training_inference(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    anchors_xy: torch.Tensor,
    device: torch.device,
    resize_size: Tuple[int, int],
    out_path: Union[str, Path],
    grid_shape=(3, 3),
):
    """
    Performs in-training qualitative inference and saves a visualization of predictions vs. ground truth.

    This function samples a grid of images from the validation loader, performs forward inference,
    decodes and filters predictions, and overlays the predicted and ground-truth oriented bounding
    boxes on the images for visual inspection.

    Ground-truth boxes are drawn in blue (with edge 0→1 in red), and predictions in green (with edge 0→1 in orange).

    Args:
        model (nn.Module): The model to evaluate.
        val_loader (DataLoader): Validation dataloader.
        anchors_xy (Tensor): Anchor boxes in vertex format (N, 8).
        device (torch.device): Target device for inference.
        resize_size (Tuple[int, int]): Size used to resize input images (W, H).
        run_name (str): Prefix name used to save the output visualization.
        out_path (Union[str, Path]): Path to save the output image.
        grid_shape (Tuple[int, int]): Grid shape for visual output (rows, cols).

    Returns:
        None. Saves a .jpg image showing model predictions vs ground truth.
    """
    model.eval()
    rows, cols = grid_shape
    max_samples = rows * cols
    samples = []

    with torch.inference_mode():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            outs = infer_with_rotated_nms(model, imgs, anchors_xy, resize_size)

            for b in range(imgs.size(0)):
                if len(samples) >= max_samples:
                    break

                # Get valid GT polygons and labels
                valid = batch["target"]["valid_mask"][b]
                gt_poly = batch["target"]["boxes"][b][valid].cpu()
                gt_lbl = batch["target"]["class_idx"][b][valid].cpu().numpy()
                samples.append((imgs[b].cpu(), outs[b], gt_poly, gt_lbl))

            if len(samples) >= max_samples:
                break

    # Set up subplot grid
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for ax, (img_t, pred, gt_poly, gt_lbl) in zip(axes, samples):
        ax.imshow(denormalize_image(img_t))  # Restore pixel values to [0,1] range
        ax.axis("off")

        # Plot ground truth polygons in blue, with edge 0→1 in red
        for poly, lbl in zip(gt_poly, gt_lbl):
            pts = poly.view(4, 2).numpy()
            ax.add_patch(
                MplPolygon(pts, closed=True, fill=False, edgecolor="blue", linewidth=2)
            )
            ax.plot(
                [pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]], color="red", linewidth=2
            )

        # Plot predicted polygons in green, with edge 0→1 in orange
        p_polys = pred["polygons"].cpu()
        p_scores = pred["scores"].cpu().numpy()
        p_lbls = pred["labels"].cpu().numpy().astype(int)
        for poly, lbl, sc in zip(p_polys, p_lbls, p_scores):
            pts = poly.view(4, 2).numpy()
            ax.add_patch(
                MplPolygon(
                    pts,
                    closed=True,
                    fill=False,
                    edgecolor="green",
                    linewidth=1.5,
                    linestyle="--",
                )
            )
            ax.plot(
                [pts[0, 0], pts[1, 0]],
                [pts[0, 1], pts[1, 1]],
                color="orange",
                linewidth=2,
            )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    model.train()
    print(f"[INFO] In-training inference saved to {out_path}")


def load_checkpoint_for_resuming(
    model: nn.Module, checkpoint_path: str, device: torch.device
) -> None:
    """
    Loads model weights from a saved checkpoint to resume training.

    Args:
        model (nn.Module): The model to load weights into.
        checkpoint_path (str): Path to the saved checkpoint file.
        device (torch.device): Device to map the checkpoint (e.g., CPU or GPU).
    """
    print(f"[INFO] Loading model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Remove _orig_mod. prefix if it exists (from torch.compile)
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        stripped = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                stripped[k[len("_orig_mod.") :]] = v
            else:
                stripped[k] = v
        state_dict = stripped

    model.load_state_dict(state_dict)
    print("[INFO] Checkpoint successfully loaded into model. Ready to resume training.")
