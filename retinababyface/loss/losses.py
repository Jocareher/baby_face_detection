from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import match_anchors_to_targets, decode_vertices, probiou, xyxyxyxy2xywhr


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
        alpha: float | list[float] = 1.0,
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
            losses.append((1.0 - iou).mean())

        if len(losses) == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred_obbs.device)

        return torch.stack(losses).mean()


class MultiTaskLoss(nn.Module):
    """
    Combines classification, OBB regression, and angle regression into a single multitask loss.

    Total Loss:
        L_total = lambda_cls * L_focal + lambda_obb * L_iou + lambda_rot * L_angle

    Args:
        lambda_cls (float): Weight for classification loss.
        lambda_obb (float): Weight for OBB regression loss.
        lambda_rot (float): Weight for rotation angle loss.
    """

    def __init__(
        self, lambda_cls: float = 1.0, lambda_obb: float = 1.0, lambda_rot: float = 0.5
    ):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.obb_loss = OBBLoss()
        self.rot_loss = RotationLoss()
        self.lambda_cls, self.lambda_obb, self.lambda_rot = (
            lambda_cls,
            lambda_obb,
            lambda_rot,
        )

    def forward(
        self,
        preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        anchors_xy: torch.Tensor,
        anchors_xywhr: torch.Tensor,
        image_sizes: List[Tuple[int, int]],
    ):
        """
        Args:
            preds (Tuple): Tuple of model outputs: (logits, deltas, angles)
            targets (Dict): Dictionary with GT boxes, angles and class indices
            anchors_xy (torch.Tensor): Anchor vertices (B, N, 8)
            anchors_xywhr (torch.Tensor): Anchor OBBs in xywhr format (N, 5)
            image_sizes (List[Tuple[int,int]]): Sizes of each image in the batch

        Returns:
            Tuple[torch.Tensor, float, float, float]:
                - Total loss
                - Classification loss (float)
                - OBB loss (float)
                - Angle regression loss (float)
        """
        logits, deltas, pred_angles = preds  # (B,N,C), (B,N,8), (B,N,1)
        B, N, _ = logits.shape

        loss_cls = loss_obb = loss_rot = 0.0
        valid_batches = 0

        for b in range(B):
            # 1 Anchor-to-GT assignment
            pos_mask, best_gt = match_anchors_to_targets(
                anchors_xywhr,
                targets["boxes"][b],
                targets["angle"][b].squeeze(-1),
                image_sizes[b],
            )

            # 2 Classification loss (all anchors)
            tgt_cls = torch.full((N,), 5, dtype=torch.long, device=logits.device)
            tgt_cls[pos_mask] = targets["class_idx"][b][best_gt[pos_mask]]
            loss_cls += self.focal_loss(logits[b], tgt_cls)

            # 3 OBB + Angle loss (only positive anchors)
            if pos_mask.any():
                # Only consider positive anchors
                valid_batches += 1
                # Extract the positive anchors
                idx = best_gt[pos_mask]

                # Compute the OBB loss
                # Note: pred_xy is in xywhr format
                # Convert deltas to absolute vertex coordinates
                # Note: deltas are in normalized coordinates
                # Decode the deltas to get the predicted vertices
                # Note: anchors_xy are in pixel coordinates
                pred_xy = decode_vertices(
                    deltas[b][pos_mask], anchors_xy[b][pos_mask], image_sizes[b]
                )

                loss_obb += self.obb_loss(
                    pred_xy.unsqueeze(0),
                    targets["boxes"][b][idx].unsqueeze(0),
                    pred_angles[b][pos_mask].unsqueeze(0),
                    targets["angle"][b][idx].unsqueeze(0),
                    [image_sizes[b]],
                    valid_mask=torch.ones(
                        1, idx.numel(), dtype=torch.bool, device=logits.device
                    ),
                )

                loss_rot += self.rot_loss(
                    pred_angles[b][pos_mask], targets["angle"][b][idx]
                )

        # Final aggregation
        loss_cls /= B
        if valid_batches == 0:
            # No valid batches, return only classification loss
            total = self.lambda_cls * loss_cls
            return total, loss_cls.item(), 0.0, 0.0

        loss_obb /= valid_batches
        loss_rot /= valid_batches
        total = (
            self.lambda_cls * loss_cls
            + self.lambda_obb * loss_obb
            + self.lambda_rot * loss_rot
        )
        return total, loss_cls.item(), loss_obb.item(), loss_rot.item()
