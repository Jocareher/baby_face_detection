from typing import List, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    match_anchors_to_targets,
    decode_vertices,
    probiou,
    xyxyxyxy2xywhr,
    encode_vertices,
)
import config


class FocalLoss(nn.Module):
    """
    Focal Loss for single-label multi-class classification (softmax).

    L = - α_t * (1 - p_t)^γ * log(p_t)
    where p_t is the probability assigned to the correct class.

    Args:
        alpha (float or list[float]): Balancing factor α. If float, same for all classes.
                                        If list, must have length = num_classes.
        gamma (float): Focusing parameter γ ≥ 0.
        ignore_index (int): Target label to be ignored (does not contribute to the loss).
        reduction (str): 'none' | 'mean' | 'sum'.
    """

    def __init__(
        self,
        alpha: Union[float, List[float]] = 1.0,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = torch.tensor([alpha], dtype=torch.float32)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape (..., C) with unnormalized scores.
            targets: Tensor of shape (...) with integers in [0, C-1] or = ignore_index.
        """
        # flatten all dimensions except the classes dimension
        orig_shape = logits.shape
        C = logits.shape[-1]
        logits = logits.view(-1, C)  # (N, C)
        targets = targets.view(-1)  # (N,)

        # mask of valid elements
        valid = targets != self.ignore_index  # (N,)
        logits = logits[valid]
        targets = targets[valid]

        if logits.numel() == 0:
            # nothing to compute
            return torch.tensor(0.0, device=logits.device)

        # softmax + log
        log_probs = F.log_softmax(logits, dim=-1)  # (M, C)
        probs = log_probs.exp()  # (M, C)

        # gather log_prob and prob of the correct class
        targets_unsq = targets.unsqueeze(1)  # (M,1)
        log_pt = log_probs.gather(1, targets_unsq).squeeze(1)  # (M,)
        pt = probs.gather(1, targets_unsq).squeeze(1)  # (M,)

        # α_t: weight per example based on its class
        if self.alpha.numel() == 1:
            at = self.alpha.to(logits.device)
        else:
            at = self.alpha.to(logits.device).gather(0, targets)  # (M,)

        # focal calculation
        loss = -at * (1 - pt).pow(self.gamma) * log_pt  # (M,)

        # reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            # reconstruct original shape with zeros where ignore_index
            out = torch.zeros(valid.shape, device=loss.device)
            out[valid] = loss
            return out.view(*orig_shape[:-1])


class RotationLoss(nn.Module):
    """
    Module for calculating the rotation loss between predicted and ground truth angles.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_angles: torch.Tensor,
        gt_angles: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calculates the rotation loss.

        The loss function is defined as:
        L_rot = 1 - cos(pred_angle - gt_angle)

        This loss penalizes the difference between predicted and ground truth angles,
        with a higher loss for larger differences.

        Args:
            pred_angles (torch.Tensor): Predicted angles tensor of shape (B, N, 1).
            gt_angles (torch.Tensor): Ground truth angles tensor of shape (B, N, 1).
            valid_mask (torch.Tensor, optional): Valid mask tensor of shape (B, N) to exclude certain predictions from loss calculation.

        Returns:
            torch.Tensor: The mean rotation loss.
        """
        # Both tensors are of shape (B, N, 1)
        diff = pred_angles - gt_angles  # (…,1)
        loss = 1 - torch.cos(diff)
        if valid_mask is not None:
            loss = loss[
                valid_mask
            ]  # Apply valid mask to exclude certain predictions from loss.
        return loss.mean()  # Return the mean rotation loss.


class OBBLoss(nn.Module):
    """
    Computes loss between predicted and ground-truth Oriented Bounding Boxes (OBBs)
    using probabilistic IoU as a similarity metric.

    Note:
        - Boxes are expected in vertex format (xyxyxyxy) and angles in radians.
        - Internally converts boxes to xywhr format.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_obbs: torch.Tensor,  # (B, N, 8)
        gt_obbs: torch.Tensor,  # (B, N, 8)
        angles_pred: torch.Tensor,  # (B, N, 1)
        gt_angles: torch.Tensor,  # (B, N, 1)
        image_size: List[Tuple[int, int]],
        valid_mask: torch.Tensor = None,  # (B, N) boolean mask for valid anchors
    ) -> torch.Tensor:
        """
        Args:
            pred_obbs (torch.Tensor): Predicted vertices, shape (B, N, 8).
            gt_obbs (torch.Tensor): Ground truth vertices, shape (B, N, 8).
            angles_pred (torch.Tensor): Predicted angles in radians, shape (B, N, 1).
            gt_angles (torch.Tensor): Ground truth angles in radians, shape (B, N, 1).
            image_size (List[Tuple[int, int]]): List of image sizes (W, H) for each sample in batch.
            valid_mask (torch.Tensor, optional): Boolean mask for valid anchors. Defaults to None.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        B, N, _ = pred_obbs.shape
        losses = []

        for b in range(B):
            W, H = image_size[b]

            # Convert both boxes to xywhr format
            pred_xywhr = xyxyxyxy2xywhr(
                pred_obbs[b], angles_pred[b].squeeze(-1), (W, H)
            )
            gt_xywhr = xyxyxyxy2xywhr(gt_obbs[b], gt_angles[b].squeeze(-1), (W, H))

            # Compute the probabilistic IoU
            # Note: the function is vectorized and computes all pairwise IoUs
            # between the predicted and ground truth boxes
            if valid_mask is not None:
                # Apply the valid mask to filter out invalid predictions
                mask = valid_mask[b]
                # Ensure the mask is applied to both pred and gt boxes
                pred_xywhr = pred_xywhr[mask]
                gt_xywhr = gt_xywhr[mask]

            # If there are no valid predictions, skip this batch
            if pred_xywhr.numel() == 0:
                continue
            # Compute the probabilistic IoU
            # Note: probiou returns a matrix of shape (N, M) where N is the number of gt boxes
            # and M is the number of pred boxes
            # We take the mean IoU across all gt boxes for this batch
            # and all pred boxes
            iou = probiou(pred_xywhr, gt_xywhr)
            # iou is of shape (N, M) where N is the number of gt boxes
            # and M is the number of pred boxes
            # We want to compute the mean IoU for each gt box
            # and take the mean across all gt boxes
            match_iou = iou.diag()
            losses.append((1.0 - match_iou).mean())

        if len(losses) == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred_obbs.device)

        return torch.stack(losses).mean()


class SmoothL1OBBLoss(nn.Module):
    """
    Computes Smooth L1 loss between predicted OBB deltas and encoded ground truth deltas.

    The loss is applied directly on the 8-point vertex offsets, normalized with respect
    to the anchor box diagonal, as defined in the encode_vertices() function.
    """

    def __init__(self, beta: float = 1.0, reduction: str = "mean"):
        """
        Args:
            beta (float): Transition point between L1 and L2 loss in Smooth L1.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        pred_deltas: torch.Tensor,  # (B=1, N_pos, 8) or (N_pos, 8)
        gt_xy: torch.Tensor,  # (B=1, N_pos, 8) or (N_pos, 8)
        anchors: torch.Tensor,  # (B=1, N_pos, 8) or (N_pos, 8)
    ) -> torch.Tensor:
        """
        Args:
            pred_deltas (Tensor): Predicted normalized deltas from the OBB head.
            gt_xy (Tensor): Ground truth boxes in absolute vertex coordinates.
            anchors (Tensor): Anchor boxes in absolute vertex coordinates.

        Returns:
            torch.Tensor: Smooth L1 loss between predicted and encoded deltas.
        """
        # Remove batch dimension if present
        if pred_deltas.dim() == 3 and pred_deltas.size(0) == 1:
            pred = pred_deltas.squeeze(0)
            gt = gt_xy.squeeze(0)
            anc = anchors.squeeze(0)
        else:
            pred, gt, anc = pred_deltas, gt_xy, anchors

        # Encode ground truth to normalized delta space
        gt_deltas = encode_vertices(gt, anc)

        # Compute Smooth L1 loss
        return F.smooth_l1_loss(
            pred, gt_deltas, beta=self.beta, reduction=self.reduction
        )


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining classification, OBB regression, and angle regression.

    The total loss is defined as:
        L_total = λ_cls * L_focal + λ_obb * L_smoothL1 + λ_rot * L_angle

    Args:
        lambda_cls (float): Weight for the classification loss.
        lambda_obb (float): Weight for the oriented bounding box regression loss.
        lambda_rot (float): Weight for the angle regression loss.
        pos_iou_thresh (float): IoU threshold to consider an anchor as positive.
        alpha (List[float]): Class-balancing weights for focal loss.
        gamma (float): Focusing parameter for focal loss.
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_obb: float = 1.0,
        lambda_rot: float = 0.5,
        pos_iou_thresh: float = config.POS_IOU_THRESH,
        alpha: List[float] = config.ALPHA,
        gamma: float = config.GAMMA,
    ):
        super().__init__()
        self.focal_loss = FocalLoss(
            alpha=alpha, gamma=gamma, ignore_index=-100, reduction="mean"
        )
        self.obb_loss = SmoothL1OBBLoss()
        self.rot_loss = RotationLoss()
        self.lambda_cls = lambda_cls
        self.lambda_obb = lambda_obb
        self.lambda_rot = lambda_rot
        self.pos_iou_thr = pos_iou_thresh

    def forward(
        self,
        preds: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor
        ],  # (logits, deltas, angles)
        targets: Dict[
            str, torch.Tensor
        ],  # GT dict: boxes, angle, class_idx, valid_mask
        anchors_xy: torch.Tensor,  # (B, N, 8) anchor vertices
        anchors_xywhr: torch.Tensor,  # (N, 5) anchors in xywhr
        image_sizes: List[Tuple[int, int]],  # (W, H) for each image
    ) -> Tuple[torch.Tensor, float, float, float]:
        """
        Computes the total multi-task loss and returns all components.

        Args:
            preds (Tuple[Tensor]): Output tuple (logits, deltas, angles) from the model.
            targets (Dict[str, Tensor]): Ground-truth information per image.
            anchors_xy (Tensor): Anchor boxes in xyxyxyxy format (B, N, 8).
            anchors_xywhr (Tensor): Anchor boxes in (cx, cy, w, h, θ) format (N, 5).
            image_sizes (List[Tuple[int, int]]): Image sizes in (width, height) format.

        Returns:
            Tuple[Tensor, float, float, float]:
                - total_loss: Combined loss tensor.
                - cls_loss (float): Classification loss.
                - obb_loss (float): Oriented bounding box regression loss.
                - rot_loss (float): Rotation angle regression loss.
        """
        logits, deltas, pred_angles = preds
        B, N, C = logits.shape

        cls_loss = 0.0
        obb_loss = 0.0
        rot_loss = 0.0
        valid_batches = 0

        for b in range(B):
            # Step 1: Match anchors to GT using IoU
            pos_mask, best_gt = match_anchors_to_targets(
                anchors_xywhr,
                targets["boxes"][b],
                targets["angle"][b].squeeze(-1),
                image_sizes[b],
                iou_thr=self.pos_iou_thr,
            )
            # print(f"batch {b}: pos={pos_mask.sum().item()} / {pos_mask.numel()}")

            # Step 2: Classification loss using focal loss
            tgt_cls = torch.full((N,), 5, dtype=torch.long, device=logits.device)
            tgt_cls[pos_mask] = targets["class_idx"][b][best_gt[pos_mask]]
            cls_loss += self.focal_loss(logits[b], tgt_cls)

            # Step 3: OBB regression loss (only for positives)
            if pos_mask.any():
                valid_batches += 1
                idx = best_gt[pos_mask]
                pred_deltas = deltas[b][pos_mask]
                gt_xy = targets["boxes"][b][idx]
                anc_xy = anchors_xy[b][pos_mask]

                obb_loss += self.obb_loss(
                    pred_deltas.unsqueeze(0), gt_xy.unsqueeze(0), anc_xy.unsqueeze(0)
                )

                # Step 4: Angle regression loss
                rot_loss += self.rot_loss(
                    pred_angles[b][pos_mask], targets["angle"][b][idx]
                )

        # Normalize losses over batch
        cls_loss /= B
        if valid_batches == 0:
            total_loss = self.lambda_cls * cls_loss
            return total_loss, cls_loss.item(), 0.0, 0.0

        obb_loss /= valid_batches
        rot_loss /= valid_batches

        total_loss = (
            self.lambda_cls * cls_loss
            + self.lambda_obb * obb_loss
            + self.lambda_rot * rot_loss
        )

        return total_loss, cls_loss.item(), obb_loss.item(), rot_loss.item()
