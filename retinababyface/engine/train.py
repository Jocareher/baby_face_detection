import time
import csv
import os
import random
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import tqdm.auto as tqdm_auto
from torch.nn import functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

tqdm = tqdm_auto.tqdm  # Use tqdm.auto for better compatibility with Jupyter notebooks

from models.anchors import AnchorGeneratorOBB, get_feature_map_shapes
from models.retinababyface import ViTFeature2D
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
) -> torch.Tensor:
    """
    Performs Non-Maximum Suppression (NMS) for rotated bounding boxes (OBBs)
    using pairwise probabilistic IoU (pIoU) as the suppression criterion.

    Boxes with IoU ≥ threshold and lower score are suppressed.

    Args:
        boxes (Tensor): Tensor of shape (N, 5) with rotated boxes in (cx, cy, w, h, θ) format.
        scores (Tensor): Confidence scores for each box, shape (N,).
        threshold (float): IoU threshold for suppression (default: 0.45).

    Returns:
        Tensor: Indices of boxes kept after NMS, referring to the original input order.
    """
    # Sort boxes by descending score
    order = scores.argsort(descending=True)
    boxes = boxes[order]
    scores = scores[order]

    # Compute full pairwise IoU matrix (NxN)
    ious = batch_probiou(boxes, boxes)
    mask = ious >= threshold
    mask.fill_diagonal_(False)  # Avoid suppressing self

    # Initialize all boxes as "to keep"
    keep = torch.ones_like(scores, dtype=torch.bool)

    # For each box, suppress any lower-scored box with high IoU
    for i in range(scores.size(0)):
        if keep[i]:
            keep &= ~(mask[i] & (scores < scores[i]))

    # Return the indices relative to the original input
    return order[keep]


def infer_with_rotated_nms(
    model: nn.Module,
    images: torch.Tensor,  # (B, 3, H, W)
    anchors_xy: torch.Tensor,  # (N, 8) anchor corners in xyxyxyxy
    image_size: Tuple[int, int],  # (W, H)
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    pre_nms_topk: int = 1000,
    max_det: int = 300,
) -> List[Dict[str, torch.Tensor]]:
    """
    Performs inference using a RetinaBabyFace-like model and applies rotated NMS.

    Args:
        model (nn.Module): Model that outputs classification logits, vertex deltas, and angles.
        images (Tensor): Batch of input images, shape (B, 3, H, W).
        anchors_xy (Tensor): Anchor polygons in (N, 8) format (4 corners).
        image_size (Tuple[int, int]): Size of input images (W, H).
        conf_thres (float): Confidence threshold to filter low-score predictions.
        iou_thres (float): IoU threshold for rotated NMS.
        pre_nms_topk (int): Max number of top scoring predictions before NMS.
        max_det (int): Max number of final predictions per image.

    Returns:
        List[Dict[str, Tensor]]: List of length B with dicts per image containing:
            - 'boxes':    (M, 5) boxes in (cx, cy, w, h, θ) format
            - 'scores':   (M,)   confidence scores
            - 'labels':   (M,)   class indices (float)
            - 'polygons': (M, 8) 4-corner polygons for visualization
    """
    B = images.size(0)
    logits, deltas, pred_angles = model(images)  # → (B, N, C), (B, N, 8), (B, N, 1)
    prob = F.softmax(logits, dim=-1)
    outputs = []

    for b in range(B):
        # Get top class and score per anchor
        scores_b, labels_b = prob[b].max(-1)

        # Filter out background and low-confidence detections
        keep_mask = (labels_b != 5) & (scores_b > conf_thres)
        if not keep_mask.any():
            outputs.append(
                {
                    "boxes": torch.zeros((0, 5), device=images.device),
                    "scores": torch.zeros((0,), device=images.device),
                    "labels": torch.zeros((0,), device=images.device),
                    "polygons": torch.zeros((0, 8), device=images.device),
                }
            )
            continue

        # 1) Get original indices of kept proposals
        idxs = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)  # (N_valid,)

        # 2) Select top-K proposals by score
        scores_k = scores_b[idxs]  # (N_valid,)
        K = min(pre_nms_topk, scores_k.size(0))
        topk_scores, topk_inds = scores_k.topk(K, sorted=True)  # (K,)
        sel = idxs[topk_inds]  # (K,) indices w.r.t. full anchor set

        # 3) Decode selected proposals into polygons
        verts = decode_vertices(
            deltas[b][sel],  # (K, 8)
            anchors_xy[sel],  # (K, 8)
            image_size,
            use_diag=True,
        )  # → (K, 8)

        # 4) Convert to (cx, cy, w, h, θ) format for NMS
        xywhr = xyxyxyxy2xywhr(
            verts, pred_angles[b][sel].squeeze(-1), image_size  # (K,)
        )  # (K, 5)

        # 5) Apply rotated NMS
        keep_nms = nms_rotated(xywhr, topk_scores, threshold=iou_thres)
        keep_nms = keep_nms[:max_det]

        # 6) Map back to original indices in logits/predictions
        sel_final = sel[keep_nms]  # Final selection

        # 7) Prepare output
        outputs.append(
            {
                "boxes": xywhr[keep_nms],  # (M, 5)
                "scores": topk_scores[keep_nms],  # (M,)
                "labels": labels_b[sel_final].float(),  # (M,)
                "polygons": verts[keep_nms],  # (M, 8)
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
    num_classes: int = 6,
) -> float:
    """
    Computes the mean Average Precision (mAP) for rotated bounding boxes across all non-background classes.

    This implementation uses a Pascal VOC-style 11-point interpolation method and supports
    rotated bounding boxes in (cx, cy, w, h, θ) format. IoU is computed using a probabilistic
    IoU function (batch_probiou), and background class is excluded from mAP computation.

    Args:
        all_pred_boxes (List[Tensor]): List of (N_i, 5) tensors with predicted boxes for each image.
        all_pred_scores (List[Tensor]): List of (N_i,) tensors with confidence scores per prediction.
        all_pred_labels (List[Tensor]): List of (N_i,) tensors with predicted class indices.
        all_gt_boxes (List[Tensor]): List of (M_i, 5) tensors with ground truth boxes for each image.
        all_gt_labels (List[Tensor]): List of (M_i,) tensors with ground truth class indices.
        iou_thr (float): IoU threshold to consider a prediction a true positive.
        num_classes (int): Total number of classes including background.

    Returns:
        float: The computed mean Average Precision (mAP) over all foreground classes.
    """
    # 0) If there are no predictions at all, return mAP = 0.0
    if len(all_pred_scores) == 0:
        return 0.0

    device = all_pred_scores[0].device
    APs: List[torch.Tensor] = []

    # Define which class is the background class
    bg_idx = num_classes - 1

    # Compute AP for each class (excluding background)
    for c in range(num_classes):
        if c == bg_idx:
            continue

        # 1) Collect all predictions for class `c` across all images
        preds = []
        for img_i in range(len(all_pred_boxes)):
            mask = all_pred_labels[img_i] == c
            for box, score in zip(
                all_pred_boxes[img_i][mask], all_pred_scores[img_i][mask]
            ):
                preds.append({"img": img_i, "box": box, "score": score})
        preds.sort(
            key=lambda x: x["score"].item(), reverse=True
        )  # Sort descending by score

        # 2) Collect ground truth boxes for class `c` per image
        gt_per_img = {
            i: all_gt_boxes[i][all_gt_labels[i] == c].clone().to(device)
            for i in range(len(all_gt_boxes))
        }
        npos = sum(
            len(v) for v in gt_per_img.values()
        )  # Total number of GT boxes for this class

        # If no GT boxes exist for this class
        if npos == 0:
            APs.append(torch.tensor(1.0 if not preds else 0.0, device=device))
            continue

        # 3) Initialize true positive (TP) and false positive (FP) vectors
        tp = torch.zeros(len(preds), device=device)
        fp = torch.zeros(len(preds), device=device)
        detected = {
            i: torch.zeros(len(gt_per_img[i]), dtype=torch.bool, device=device)
            for i in gt_per_img
        }

        # 4) Match predictions to GT using probabilistic IoU
        for idx, p in enumerate(preds):
            img_i = p["img"]
            gt_boxes = gt_per_img[img_i]

            # No GTs in this image for this class → count as FP
            if gt_boxes.numel() == 0:
                fp[idx] = 1
                continue

            # Compute IoUs between prediction and all GTs for this image
            ious = batch_probiou(p["box"].unsqueeze(0), gt_boxes)  # shape: (1, M)
            best_iou, best_j = ious[0].max(dim=0)

            if best_iou >= iou_thr and not detected[img_i][best_j]:
                tp[idx] = 1
                detected[img_i][best_j] = True  # Mark GT as matched
            else:
                fp[idx] = 1

        # 5) Compute precision and recall curves
        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        recall = tp_cum / (npos + 1e-6)
        precision = tp_cum / (tp_cum + fp_cum + 1e-6)

        # 6) Compute 11-point interpolated average precision
        ap = torch.tensor(0.0, device=device)
        for t in torch.linspace(0, 1, 11, device=device):
            mask = recall >= t
            if mask.any():
                ap += precision[mask].max() / 11.0
        APs.append(ap)

    # If no classes were processed (edge case), return 0
    if len(APs) == 0:
        return 0.0

    # Return the mean of all per-class APs
    return torch.stack(APs).mean().item()


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
    base_size: float,
    base_ratio: float,
    scale_factors: List[float],
    ratio_factors: List[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates oriented anchor boxes (OBBs) for the training stage of the model, based on the
    feature map resolutions and the anchor generation configuration.

    This function infers the feature map sizes from the model's architecture given a specified input
    resolution. It then uses an OBB anchor generator to create rotated anchor boxes in both
    vertex-based (xyxyxyxy) and parameterized (xywhr) formats. A visual preview of the anchors
    is optionally saved as an image.

    Args:
        model (nn.Module): The model from which to extract the feature map shapes.
        resize_size (Tuple[int, int]): The target input image size as (width, height).
        device (torch.device): Device on which tensors are created (CPU or GPU).
        base_size (float): Base size of the anchor generator (scales the default anchors).
        base_ratio (float): Base aspect ratio of the anchors.
        scale_factors (List[float]): List of scale multipliers to generate anchors of various sizes.
        ratio_factors (List[float]): List of aspect ratio multipliers to create anchors of different shapes.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - anchors_xy (torch.Tensor): Anchor boxes in vertex format (N, 8).
            - anchors_xywhr (torch.Tensor): Anchor boxes in (cx, cy, w, h, angle) format (N, 5).
    """

    # Unpack target input size (W, H)
    H, W = resize_size[1], resize_size[0]

    # Get the output feature map shapes for each FPN level from the model
    feature_shapes = get_feature_map_shapes(model, input_shape=(1, 3, H, W))

    # Compute the stride (downsampling factor) for each feature map
    strides = [int(round(H / h)) for (h, _w) in feature_shapes]

    # Initialize the oriented bounding box anchor generator
    anchor_gen = AnchorGeneratorOBB(
        base_size=base_size,
        base_ratio=base_ratio,
        scale_factors=scale_factors,
        ratio_factors=ratio_factors,
        angles=config.ANGLES,  # List of fixed rotation angles
    )

    # Generate anchor boxes in xyxyxyxy format (8 points per box)
    anchors_xy = anchor_gen.generate_anchors(
        feature_map_shapes=feature_shapes,
        strides=strides,
        device=device,
    )

    # Convert the anchors to parameterized (cx, cy, w, h, angle) format
    zeros = torch.zeros(len(anchors_xy), device=device)  # No angle during generation
    anchors_xywhr = xyxyxyxy2xywhr(anchors_xy, zeros, (W, H))

    # Optionally save a preview of a sample of anchors
    preview_path = "anchors_preview.jpg"
    if not os.path.exists(preview_path):
        all_anc = anchors_xy.cpu().numpy()  # (N, 8)
        K = min(200, all_anc.shape[0])  # Sample K anchors for visualization
        idxs = random.sample(range(all_anc.shape[0]), K)

        # Use HSV colormap for diverse colors
        cmap = plt.cm.get_cmap("hsv", K)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
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
        plt.savefig(preview_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Anchor preview saved to {preview_path}")

    return anchors_xy, anchors_xywhr


def create_optimizer(
    which_optimizer: str, model: nn.Module, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    """
    Creates an optimizer for the model.

    Args:
        which_optimizer (str): The optimizer to use ('ADAM' or 'SGD').
        model (nn.Module): The model to optimize.
        learning_rate (float): The learning rate.
        weight_decay (float): The weight decay.

    Returns:
        torch.optim.Optimizer: The optimizer.

    Raises:
        ValueError: If the optimizer is not 'ADAM' or 'SGD'.
    """
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
    else:
        raise ValueError("The optimizer must be 'ADAM' or 'SGD'.")


def create_scheduler(
    which_scheduler: Optional[str],
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    epochs: int,
    train_dataloader: DataLoader,
) -> Optional[lr_scheduler._LRScheduler]:
    """
    Creates a learning rate scheduler.

    Args:
        which_scheduler (Optional[str]): The scheduler to use ('ReduceLR', 'OneCycle', 'Cosine', or None).
        optimizer (torch.optim.Optimizer): The optimizer.
        learning_rate (float): The initial learning rate.
        epochs (int): The number of epochs.
        train_dataloader (DataLoader): The training data loader.

    Returns:
        Optional[lr_scheduler._LRScheduler]: The learning rate scheduler, or None if no scheduler is used.

    Raises:
        ValueError: If the scheduler is not 'ReduceLR', 'OneCycle', or 'Cosine'.
    """
    if which_scheduler == "ReduceLR":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=5, min_lr=1e-5
        )
    elif which_scheduler == "OneCycle":
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,
            epochs=epochs,
            steps_per_epoch=len(train_dataloader),
        )
    elif which_scheduler == "Cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    elif which_scheduler is None:
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
    anchors: torch.Tensor,
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
        Tuple[float, float, float, float, float]: Average total loss, average class loss, average OBB loss, average angular loss, and current learning rate.
    """
    model.train()  # Set the model to training mode.
    total_loss_sum = 0.0
    class_loss_sum = 0.0
    obb_loss_sum = 0.0
    angular_loss_sum = 0.0
    total_batches = 0

    for batch in train_dataloader:
        images = batch["image"].to(device)  # Move images to the device.
        targets_raw = batch["target"]
        targets = build_multitask_targets(
            targets_raw, device
        )  # Process targets for multi-task learning.

        optimizer.zero_grad()  # Zero the gradients.
        anchors_xy, anchors_xywhr = anchors  #
        batch_anchors = anchors_xy.unsqueeze(0).repeat(images.size(0), 1, 1)
        pred = model(images)  # Forward pass.
        image_sizes = [(images.shape[3], images.shape[2])] * images.size(
            0
        )  # [(W, H), ...]
        loss, loss_class, loss_obb, loss_angle = loss_fn(
            pred, targets, batch_anchors, anchors_xywhr, image_sizes
        )  # Calculate loss.

        loss.backward()  # Backward pass.

        if clip_value is not None:
            if grad_clip_mode == "Norm":
                clip_grad_norm_(
                    model.parameters(), clip_value
                )  # Clip gradients by norm.
            elif grad_clip_mode == "Value":
                clip_grad_value_(
                    model.parameters(), clip_value
                )  # Clip gradients by value.

        optimizer.step()  # Update model parameters.

        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()  # Update learning rate if using OneCycleLR scheduler.

        total_loss_sum += loss.item()
        class_loss_sum += loss_class
        obb_loss_sum += loss_obb
        angular_loss_sum += loss_angle
        total_batches += 1

    current_lr = optimizer.param_groups[0]["lr"]  # Get current learning rate.
    avg_total_loss = total_loss_sum / total_batches
    avg_class_loss = class_loss_sum / total_batches
    avg_obb_loss = obb_loss_sum / total_batches
    avg_angular_loss = angular_loss_sum / total_batches

    return avg_total_loss, avg_class_loss, avg_obb_loss, avg_angular_loss, current_lr


def val_step(
    model: nn.Module,
    val_dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    anchors: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[float, float, float, float, float]:
    """
    Runs one full evaluation loop on the test dataset, computing predictions, losses, and rotated mAP.

    Args:
        model (nn.Module): The trained model to evaluate.
        val_dataloader (DataLoader): DataLoader that provides batches of test data.
        loss_fn (nn.Module): Multi-task loss function that returns total and sub-losses.
        device (torch.device): The device (CPU/GPU) on which computation will be performed.
        anchors (Tuple[Tensor, Tensor]): A tuple containing:
            - anchors_xy (Tensor): Tensor of base anchor vertices (N, 8).
            - anchors_xywhr (Tensor): Tensor of anchors in (cx, cy, w, h, θ) format (N, 5).

    Returns:
        Tuple[float, float, float, float, float]: A tuple containing:
            - avg_loss (float): Average total loss across all test batches.
            - avg_class_loss (float): Average classification loss.
            - avg_obb_loss (float): Average oriented bounding box (OBB) loss.
            - avg_angular_loss (float): Average angular prediction loss.
            - mAP (float): Rotated mean Average Precision using 11-point interpolation.
    """
    model.eval()  # Switch model to evaluation mode (no dropout, batchnorm is fixed).

    # Initialize accumulators for different losses
    total_loss = 0.0
    class_loss_sum = 0.0
    obb_loss_sum = 0.0
    angular_loss_sum = 0.0
    total_batches = 0

    # Prepare containers to collect predictions and ground truths
    all_pred_boxes, all_pred_scores, all_pred_labels = [], [], []
    all_gt_boxes, all_gt_labels = [], []

    # Disable gradient computation for faster inference and lower memory usage
    with torch.inference_mode():
        for batch in val_dataloader:
            # Move image tensors to the specified device
            images = batch["image"].to(device)

            # Extract raw targets and prepare them for multitask loss
            targets_raw = batch["target"]
            targets = build_multitask_targets(targets_raw, device)

            # Unpack anchors for decoding and loss computation
            anchors_xy, anchors_xywhr = anchors
            batch_anchors = anchors_xy.unsqueeze(0).repeat(images.size(0), 1, 1)

            # Run inference + NMS to get filtered predictions
            outputs = infer_with_rotated_nms(
                model,
                images,
                anchors_xy,
                image_size=(images.shape[3], images.shape[2]),  # (W, H)
            )

            # Accumulate predictions and ground truths for evaluation
            for b, out in enumerate(outputs):
                all_pred_boxes.append(out["boxes"])  # Predicted rotated boxes
                all_pred_scores.append(out["scores"])  # Prediction scores
                all_pred_labels.append(out["labels"])  # Predicted class labels

                # Select valid targets and convert to (cx, cy, w, h, θ)
                keep = targets["valid_mask"][b]
                gt_xywhr = xyxyxyxy2xywhr(
                    targets["boxes"][b][keep],
                    targets["angle"][b][keep].squeeze(-1),
                    (images.shape[3], images.shape[2]),
                )
                all_gt_boxes.append(gt_xywhr)
                all_gt_labels.append(targets["class_idx"][b][keep])

            # Forward pass for raw output (needed for computing loss)
            pred = model(images)

            # Generate image size tuples (one per image in batch)
            image_sizes = [(images.shape[3], images.shape[2])] * images.size(0)

            # Compute multitask loss and individual components
            loss, loss_class, loss_obb, loss_angle = loss_fn(
                pred, targets, batch_anchors, anchors_xywhr, image_sizes
            )

            # Accumulate loss values
            total_loss += loss.item()
            class_loss_sum += loss_class
            obb_loss_sum += loss_obb
            angular_loss_sum += loss_angle
            total_batches += 1

    # Compute average losses across all batches
    avg_loss = total_loss / total_batches
    avg_class_loss = class_loss_sum / total_batches
    avg_obb_loss = obb_loss_sum / total_batches
    avg_angular_loss = angular_loss_sum / total_batches

    # Compute rotated mean Average Precision (mAP) using Pascal VOC 11-point interpolation
    mAP = compute_map_rotated(
        all_pred_boxes,
        all_pred_scores,
        all_pred_labels,
        all_gt_boxes,
        all_gt_labels,
        iou_thr=0.5,
        num_classes=6,
    )

    return avg_loss, avg_class_loss, avg_obb_loss, avg_angular_loss, mAP


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
    obb_stats_by_size: Optional[Dict[Tuple[int, int], Dict[str, float]]] = None,
    grid_shape: Tuple[int, int] = (3, 3),
) -> Dict[str, List[float]]:
    """
    Trains the model and optionally records metrics.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the testing dataset.
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
        obb_stats_by_size (Dict[Tuple[int, int], Dict[str, float]], optional): Precomputed OBB statistics by size.

    Returns:
        Dict[str, List[float]]: Dictionary containing lists of training and testing losses.
    """

    # CSV file name for logging metrics
    csv_filename = f"{run_name}.csv"
    header = [
        "epoch",
        "train_total_loss",
        "train_class_loss",
        "train_obb_loss",
        "train_angular_loss",
        "test_total_loss",
        "test_class_loss",
        "test_obb_loss",
        "test_angular_loss",
        "test_mAP",
        "learning_rate",
        "epoch_time",
    ]
    # Create CSV file and write header
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    results = {
        "train_total_loss": [],
        "train_class_loss": [],
        "train_obb_loss": [],
        "train_angular_loss": [],
        "test_total_loss": [],
        "test_class_loss": [],
        "test_obb_loss": [],
        "test_angular_loss": [],
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
        ], "grad_clip_mode must be 'Norm' or 'Value'"  # Check valid gradient clip mode

    ## Get the resize size from the dataloader
    resize_size = get_resize_size(train_dataloader)
    base_size, base_ratio = get_base_obb_stats(resize_size, obb_stats_by_size)

    # Generate anchors for training
    anchors_xy, anchors_xywhr = generate_anchors_for_training(
        model=model,
        resize_size=resize_size,
        device=device,
        base_size=base_size,
        base_ratio=base_ratio,
        scale_factors=scale_factors,
        ratio_factors=ratio_factors,
    )
    # Convert anchors to xyxy format
    anchors_tuple = (anchors_xy, anchors_xywhr)

    start_time = time.time()
    if record_metrics:
        wandb.init(project=project, name=run_name)  # Initialize Weights & Biases.
        wandb.watch(model, loss_fn, log="all")  # Watch model and loss function.

    try:
        for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
            epoch_start = time.time()

            train_dataloader_tqdm = tqdm(
                train_dataloader, desc=f"Train {epoch+1}", leave=False
            )
            (
                train_total_loss,
                train_class_loss,
                train_obb_loss,
                train_angular_loss,
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
            )  # Perform a training step.

            val_dataloader_tqdm = tqdm(
                val_dataloader, desc=f"Test {epoch+1}", leave=False
            )
            (
                test_total_loss,
                test_class_loss,
                test_obb_loss,
                test_angular_loss,
                test_mAP,
            ) = val_step(
                model=model,
                val_dataloader=val_dataloader,
                loss_fn=loss_fn,
                device=device,
                anchors=anchors_tuple,
            )  # Perform a testing step.

            if scheduler is not None and isinstance(
                scheduler, lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step(
                    test_total_loss
                )  # Update learning rate scheduler if ReduceLROnPlateau.

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch+1} | LR: {current_lr:.6f} | Time: {epoch_time//60:.0f}m {epoch_time%60:.2f}s"
            )
            print(
                f"Train metrics | Train Loss: {train_total_loss:.4f} | Class Loss: {train_class_loss:.4f} | OBB Loss: {train_obb_loss:.4f} | Angle Loss: {train_angular_loss:.4f}"
            )
            print(
                f"Test metrics | Total Test Loss: {test_total_loss:.4f} | Class Loss: {test_class_loss:.4f} | OBB Loss: {test_obb_loss:.4f} | Angle Loss: {test_angular_loss:.4f}  | mAP: {test_mAP:.4f}"
            )

            # if device.type == "cuda":
            #     allocated_mem_MB = torch.cuda.memory_allocated(device) / (1024 ** 2)
            #     max_allocated_mem_MB = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            #     print(f"[GPU] Memory used: {allocated_mem_MB:.2f} MB | Max used this epoch: {max_allocated_mem_MB:.2f} MB")
            #     torch.cuda.reset_peak_memory_stats(device)

            if record_metrics:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_total_loss": train_total_loss,
                        "train_class_loss": train_class_loss,
                        "train_obb_loss": train_obb_loss,
                        "train_angular_loss": train_angular_loss,
                        "test_total_loss": test_total_loss,
                        "test_class_loss": test_class_loss,
                        "test_obb_loss": test_obb_loss,
                        "test_angular_loss": test_angular_loss,
                        "test_mAP": test_mAP,
                        "learning_rate": current_lr,
                        "epoch_time": epoch_time,
                    }
                )  # Log metrics to Weights & Biases.

            results["train_total_loss"].append(train_total_loss)
            results["train_class_loss"].append(train_class_loss)
            results["train_obb_loss"].append(train_obb_loss)
            results["train_angular_loss"].append(train_angular_loss)
            results["test_total_loss"].append(test_total_loss)
            results["test_class_loss"].append(test_class_loss)
            results["test_obb_loss"].append(test_obb_loss)
            results["test_angular_loss"].append(test_angular_loss)
            results["test_mAP"].append(test_mAP)

            # Write metrics to CSV file
            # Open the CSV file in append mode
            # and write the metrics for the current epoch
            with open(csv_filename, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch + 1,
                        f"{train_total_loss:.4f}",
                        f"{train_class_loss:.4f}",
                        f"{train_obb_loss:.4f}",
                        f"{train_angular_loss:.4f}",
                        f"{test_total_loss:.4f}",
                        f"{test_class_loss:.4f}",
                        f"{test_obb_loss:.4f}",
                        f"{test_angular_loss:.4f}",
                        f"{test_mAP:.4f}",
                        f"{current_lr:.5f}",
                        f"{epoch_time:.4f}",
                    ]
                )

            # every 5 epochs, save grid.jpg
            if (epoch + 1) % 2 == 0:
                in_training_inference(
                    model,
                    val_dataloader,
                    anchors_xy,
                    device,
                    resize_size,
                    run_name,
                    epoch + 1,
                    grid_shape,
                )

            if early_stopping is not None:
                early_stopping(
                    test_total_loss, model
                )  # Check early stopping condition.
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
    run_name: str,
    epoch: int,
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
        epoch (int): Current training epoch (used in filename).
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
    out_path = f"{run_name}_{epoch}.jpg"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    model.train()
    print(f"[INFO] In-training inference saved to {out_path}")
