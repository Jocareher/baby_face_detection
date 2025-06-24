from typing import List, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    match_anchors_to_targets,
    probiou,
    xyxyxyxy2xywhr,
    encode_vertices,
    decode_vertices,
)
from data_setup.augmentations import wrap_to_pi
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


class L2Loss(nn.Module):
    """ ""
    Implements the Least-Squares (L2) loss for multi-class classification tasks using one-hot targets and softmax probabilities.

    This loss function computes the squared difference between the predicted class probabilities (after softmax) and the one-hot encoded ground truth labels, optionally weighting each class by a factor `alpha`.

    The loss for each sample is computed as:
    where:
        - p_c: predicted probability for class c (after softmax)
        - y_c: one-hot encoded target for class c
        - alpha_c: weighting factor for class c (can be a scalar or a list of per-class weights)

    Args:
        alpha (float or List[float], optional): Weighting factor(s) for each class. If a single float is provided, the same weight is applied to all classes. If a list is provided, it must have length equal to the number of classes. Default is 1.0.
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the loss. Useful for masking out certain samples. Default is -100.
        reduction (str, optional): Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'.
            - 'mean': the sum of the output will be divided by the number of elements in the output.
            - 'sum': the output will be summed.
            - 'none': no reduction will be applied. Default is 'mean'.

    Shape:
        - Input: logits of shape (..., C), where C = number of classes.
        - Target: tensor of shape (...), containing class indices in [0, C-1].

    Returns:
        torch.Tensor: The computed loss. If reduction is 'none', returns a tensor of shape (N,) where N is the number of valid (non-ignored) samples.

    Notes:
        - This loss is less commonly used than cross-entropy for classification, but can be beneficial in some cases.
        - The `ignore_index` parameter allows for flexible masking of samples (e.g., for padded sequences).

    """

    def __init__(
        self,
        alpha: Union[float, List[float]] = 1.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = torch.tensor([alpha], dtype=torch.float32)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the L2 loss between softmax probabilities and one-hot targets.

        Args:
            logits (torch.Tensor): Input logits of shape (..., C).
            targets (torch.Tensor): Target class indices of shape (...).

        Returns:
            torch.Tensor: The computed loss (scalar if reduction is 'mean' or 'sum').
        """
        C = logits.shape[-1]
        p = F.softmax(logits, dim=-1)  # (..., C) - predicted probabilities

        t = targets.view(-1)  # (N,) - flatten targets
        mask = t != self.ignore_index  # (N,) - mask for valid targets
        if mask.sum() == 0:
            return logits.new_tensor(0.0)

        p = p.view(-1, C)[mask]  # (M, C) - valid predictions
        t = t[mask]  # (M,) - valid targets

        y_onehot = F.one_hot(t, C).float()  # (M, C) - one-hot encoded targets

        # α_c: class weights, broadcast to (C,)
        alpha = (
            self.alpha.to(p.device)
            if self.alpha.numel() == C
            else self.alpha.to(p.device).expand(C)
        )
        # Compute per-sample L2 loss, weighted by alpha
        loss = 0.5 * ((p - y_onehot).pow(2) * alpha).sum(dim=1)  # (M,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # "none"


class RotationLoss(nn.Module):
    """
    Computes the loss between predicted and ground-truth rotation angles.

    Supports two modes:
        - "cosine": Uses the cosine similarity loss, defined as L_rot = 1 - cos(pred_angle - gt_angle).
          This penalizes the angular difference, with a minimum at zero difference and a maximum at pi.
        - "vector": Treats angles as 2D unit vectors (sin, cos) and computes the mean squared error (MSE)
          between predicted and ground-truth vectors. This is equivalent to minimizing the squared Euclidean
          distance on the unit circle.

    Args:
        mode (str): Loss mode, either "cosine" or "vector".
            - "cosine": L_rot = 1 - cos(pred_angle - gt_angle)
            - "vector": L_rot = MSE([sin(pred), cos(pred)], [sin(gt), cos(gt)])

    Usage:
        loss_fn = RotationLoss(mode="cosine")
        loss = loss_fn(pred_angles, gt_angles, valid_mask)
    """

    def __init__(self, mode: str = "cosine"):
        super().__init__()
        assert mode in {"cosine", "vector"}, "mode must be 'cosine' or 'vector'"
        self.mode = mode

    def forward(
        self,
        pred_angles: torch.Tensor,
        gt_angles: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Computes the rotation loss between predicted and ground-truth angles.

        Args:
            pred_angles (torch.Tensor): Predicted angles, shape (..., 1).
            gt_angles (torch.Tensor): Ground-truth angles, shape (..., 1).
            valid_mask (torch.Tensor, optional): Boolean mask of shape (...) indicating valid elements.
                If provided, loss is computed only on valid entries.

        Returns:
            torch.Tensor: Scalar tensor with the mean rotation loss.
        """
        if self.mode == "cosine":
            # Cosine loss: penalizes angular difference
            diff = pred_angles - gt_angles  # (..., 1)
            loss = 1 - torch.cos(diff)
        else:  # "vector"
            # Vector loss: treat angles as points on the unit circle
            v_pred = torch.cat(
                [pred_angles.sin(), pred_angles.cos()], dim=-1
            )  # (..., 2)
            v_gt = torch.cat([gt_angles.sin(), gt_angles.cos()], dim=-1)  # (..., 2)
            loss = F.mse_loss(v_pred, v_gt, reduction="none").sum(
                -1, keepdim=True
            )  # (..., 1)

        if valid_mask is not None:
            # Apply mask to exclude invalid predictions from loss computation
            loss = loss[valid_mask]

        return loss.mean()


class OBBIoULoss(nn.Module):
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


class OBBRegressionLoss(nn.Module):
    """
    Computes either L1 or Smooth L1 loss between predicted OBB deltas and
    encoded ground-truth deltas (via encode_vertices).

    The loss is applied directly on the 8-point vertex offsets, normalized with respect
    to the anchor box diagonal, as defined in the encode_vertices() function.
    """

    def __init__(
        self,
        loss_type: str = "l1",
        beta: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            loss_type (str): "l1" for L1 loss, "smooth_l1" for Smooth-L1.
            beta (float): Transition point for Smooth-L1. Ignored if loss_type="l1".
            reduction (str): One of "none", "sum", or "mean".
        """
        super().__init__()
        assert loss_type in ("l1", "smooth_l1"), "loss_type must be 'l1' or 'smooth_l1'"
        self.loss_type = loss_type
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        pred_deltas: torch.Tensor,  # (B=1, N_pos, 8) or (N_pos, 8)
        gt_xy: torch.Tensor,  # (B=1, N_pos, 8) or (N_pos, 8)
        anchors: torch.Tensor,  # (B=1, N_pos, 8) or (N_pos, 8)
    ) -> torch.Tensor:
        # 1) Squeeze away the leading batch=1 dim, if present
        if pred_deltas.dim() == 3 and pred_deltas.size(0) == 1:
            pred = pred_deltas.squeeze(0)
            gt = gt_xy.squeeze(0)
            anc = anchors.squeeze(0)
        else:
            pred, gt, anc = pred_deltas, gt_xy, anchors

        # 2) Encode GT into the same normalized delta space
        gt_deltas = encode_vertices(gt, anc)

        # 3) Compute the selected loss
        if self.loss_type == "l1":
            return F.l1_loss(pred, gt_deltas, reduction=self.reduction)
        else:  # smooth_l1
            return F.smooth_l1_loss(
                pred, gt_deltas, beta=self.beta, reduction=self.reduction
            )


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function for RetinaFace model.
    This loss function combines the following components:
        1. Focal loss or Least Square Loss for classification (multi-class).
        2. Binary cross-entropy loss for face classification.
        3. Oriented bounding box regression loss.
        4. Rotation angle regression loss.
        5. Probabilistic IoU loss for OBBs.
    The loss function is designed to handle multiple tasks simultaneously,
    allowing the model to learn from all tasks at once.
    The loss function is defined as:
        L_cls  = CLSLoss(orient_logits, tgt_cls)
        L_face = BCE(face_logits, tgt_face)
        L_obb  = OBBRegressionLoss(pred_deltas, gt_xy, anc_xy)
        L_rot  = RotationLoss(pred_angles, gt_angles)

    The total loss is defined as:
        L_total = λ_cls * L_cls + λ_face * L_face + λ_obb * L_obb + λ_rot * L_rot

    Args:
        obb_loss_type (str): Type of OBB regression loss to use ("smooth_l1" or "l1").
        rot_loss_type (str): Type of rotation loss to use ("cosine" or "vector").
        cls_loss_type (str): Type of classification loss to use ("focal" or "ls").
        lambda_cls (float): Weight for the classification loss.
        lambda_obb (float): Weight for the oriented bounding box regression loss.
        lambda_rot (float): Weight for the angle regression loss.
        lambda_face (float): Weight for the face classification loss.
        pos_iou_thr_1 (float): IoU threshold to consider an anchor positive in stage 1.
        neg_iou_thr_1 (float): IoU threshold to consider an anchor negative in stage 1.
        pos_iou_thr_2 (float): IoU threshold to consider a provisional box positive in stage 2.
        neg_iou_thr_2 (float): IoU threshold to consider a provisional box negative in stage 2.
        alpha (List[float]): Class-balancing weights for focal loss.
        gamma (float): Focusing parameter for focal loss.
        neg_samples_ratio (int): Ratio of negative samples to positive samples for face classification.

    Note:
        - The loss function expects the model's output to be in the following format:
            (orient_logits, face_logits, deltas, angles)
        - The targets dictionary should contain the following keys:
            "boxes":      Ground truth boxes in xyxyxyxy format.
            "angle":      Ground truth angles in radians.
            "class_idx":  Class indices for each object.
            "valid_mask": Boolean mask indicating valid object positions.
        - The anchors_xy tensor should contain the anchor boxes in xyxyxyxy format.
        - The anchors_xywhr tensor should contain the anchor boxes in (cx, cy, w, h, θ) format.
        - The image_sizes list should contain the sizes of each image in the batch.
    """

    def __init__(
        self,
        obb_loss_type: str = config.OBB_LOSS_TYPE,
        rot_loss_type: str = config.ROT_LOSS_TYPE,
        cls_loss_type: str = config.CLS_LOSS_TYPE,
        lambda_cls: float = config.LAMBDA_CLS,
        lambda_obb: float = config.LAMBDA_OBB,
        lambda_rot: float = config.LAMBDA_ROT,
        lambda_face: float = config.LAMBDA_FACE,
        pos_iou_thr_1: float = config.POS_IOU_THRESH_1,
        neg_iou_thr_1: float = config.NEG_IOU_THRESH_1,
        pos_iou_thr_2: float = config.POS_IOU_THRESH_2,
        neg_iou_thr_2: float = config.NEG_IOU_THRESH_2,
        alpha: List[float] = config.ALPHA,
        gamma: float = config.GAMMA,
        neg_samples_ratio: int = config.NEG_SAMPLES_RATIO,
    ) -> None:
        super().__init__()
        if cls_loss_type == "focal":
            self.cls_loss_fn = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
        elif cls_loss_type == "ls":
            self.cls_loss_fn = L2Loss(alpha=alpha, reduction="mean")
        else:
            raise ValueError("cls_loss_type must be 'focal' or 'ls'")

        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(float(config.NEG_SAMPLES_RATIO)), reduction="mean"
        )
        self.obb_loss = OBBRegressionLoss(
            loss_type=obb_loss_type, beta=2.0, reduction="mean"
        )
        self.rot_loss = RotationLoss(mode=rot_loss_type)
        self.lambda_cls = lambda_cls
        self.lambda_obb = lambda_obb
        self.lambda_rot = lambda_rot
        self.lambda_face = lambda_face
        self.pos_iou_thr_1 = pos_iou_thr_1
        self.neg_iou_thr_1 = neg_iou_thr_1
        self.pos_iou_thr_2 = pos_iou_thr_2
        self.neg_iou_thr_2 = neg_iou_thr_2
        self.neg_samples_ratio = neg_samples_ratio

    def forward(
        self,
        preds: Tuple[
            torch.Tensor,  # orient_logits: (B, N, 5)
            torch.Tensor,  # face_logits:   (B, N, 1)
            torch.Tensor,  # deltas:        (B, N, 8)
            torch.Tensor,  # angles:        (B, N, 1)
        ],
        targets: Dict[
            str, torch.Tensor
        ],  # GT dict: boxes, angle, class_idx, valid_mask
        anchors_xy: torch.Tensor,  # (B, N, 8) anchor vertices
        anchors_xywhr: torch.Tensor,  # (N, 5) anchors in xywhr
        image_sizes: List[Tuple[int, int]],  # (W, H) for each image
    ) -> Tuple[torch.Tensor, float, float, float, float]:
        """
        Computes the total multi-task loss and returns all components:

        Args:
            preds (Tuple[Tensor]): Output tuple (orient_logits, face_logits, deltas, angles) from the model.
            targets (Dict[str, Tensor]): Ground-truth information per image.
            anchors_xy (Tensor): Anchor boxes in xyxyxyxy format (B, N, 8).
            anchors_xywhr (Tensor): Anchor boxes in (cx, cy, w, h, θ) format (N, 5).
            image_sizes (List[Tuple[int, int]]): Image sizes in (width, height) format.

        Returns:
            Tuple[Tensor, float, float, float, float]:
                - total_loss   : Combined loss tensor.
                - cls_loss     : Classification loss.
                - face_loss    : Face classification loss.
                - obb_loss     : Oriented bounding box regression loss.
                - rot_loss     : Rotation angle regression loss.
        """
        orient_logits, face_logits, deltas, pred_angles = preds
        B, N, _ = orient_logits.shape

        cls_loss = 0.0
        face_loss = 0.0
        obb_loss = 0.0
        rot_loss = 0.0
        valid_batches = 0
        num_cls_batches = 0

        for b in range(B):
            # -------------------------------------
            # Stage 1: Match anchors using IoU thresholds 1 (for face + classification)
            # -------------------------------------
            pos_mask_1, neg_mask_1, best_gt_1 = match_anchors_to_targets(
                anchors_xywhr,
                targets["boxes"][b],
                targets["angle"][b].squeeze(-1),
                image_sizes[b],
                pos_iou_thr=self.pos_iou_thr_1,
                neg_iou_thr=self.neg_iou_thr_1,
            )
            # Collect positive and negative indices for face loss
            pos_idx_1 = pos_mask_1.nonzero(as_tuple=False).squeeze(1)  # (num_pos_1,)
            neg_idx_1 = neg_mask_1.nonzero(as_tuple=False).squeeze(1)  # (num_neg_1,)

            num_pos_1 = pos_idx_1.numel()
            if num_pos_1 > 0:
                # a) Compute per-candidate negative face loss to select hard negatives
                neg_logits_all = face_logits[b][neg_idx_1]  # (num_neg_1, 1)
                neg_targets_all = torch.zeros_like(neg_logits_all)  # (num_neg_1, 1)
                with torch.no_grad():
                    per_neg_loss = F.binary_cross_entropy_with_logits(
                        neg_logits_all, neg_targets_all, reduction="none"
                    ).view(
                        -1
                    )  # (num_neg_1,)

                # b) Sort negatives by descending loss (hard negatives first)
                _, hard_order = per_neg_loss.sort(descending=True)

                # c) Select up to neg_samples_ratio * num_pos_1 hard negatives
                num_hard = min(hard_order.numel(), num_pos_1 * self.neg_samples_ratio)
                hard_neg_idx = neg_idx_1[hard_order[:num_hard]]

                # d) Combine positive and selected hard-negative indices
                sel_idx_1 = torch.cat([pos_idx_1, hard_neg_idx], dim=0)

                # e) Build face targets: 1 for positives, 0 for selected negatives
                tgt_face = pos_mask_1.float().unsqueeze(1)[
                    sel_idx_1
                ]  # (num_pos_1 + num_hard, 1)
                face_logits_sel = face_logits[b][sel_idx_1]  # (num_pos_1 + num_hard, 1)

                # Compute binary cross-entropy loss on selected anchors
                face_loss += self.bce_loss(face_logits_sel, tgt_face)

            # -------------------------------------
            # Stage 1: Classification loss (orient_logits)
            # -------------------------------------
            # Initialize all targets as ignore_index
            tgt_cls = torch.full(
                (N,),
                self.cls_loss_fn.ignore_index,
                dtype=torch.long,
                device=orient_logits.device,
            )
            if num_pos_1 > 0:
                # Assign ground-truth class labels to positive anchors
                tgt_cls[pos_idx_1] = targets["class_idx"][b][best_gt_1[pos_idx_1]]

                # ----- Focal loss with the same selection as Face-Head -----
                sel_idx_1 = torch.cat(
                    [pos_idx_1, hard_neg_idx], dim=0
                )  # Combine positive and hard-negative indices
                logits_sel = orient_logits[b][
                    sel_idx_1
                ]  # (P+H, 5) Selected logits for classification
                tgt_cls_sel = tgt_cls[
                    sel_idx_1
                ]  # (P+H,) Corresponding ground-truth labels
                cls_loss += self.cls_loss_fn(
                    logits_sel, tgt_cls_sel
                )  # Compute focal loss
                num_cls_batches += 1  # Increment batch count for classification

            # -------------------------------------
            # Stage 2: Generate provisional OBBs from stage 1 outputs
            # -------------------------------------
            if pos_mask_1.any():
                # Decode provisional vertices and angles without gradient
                with torch.no_grad():
                    pred_deltas_1 = deltas[b][pos_mask_1]  # (num_pos_1, 8)
                    anc_xy_1 = anchors_xy[b][pos_mask_1]  # (num_pos_1, 8)
                    verts_1 = decode_vertices(
                        pred_deltas_1, anc_xy_1, image_sizes[b], use_diag=True
                    )  # (num_pos_1, 8)
                    ang_1 = pred_angles[b][pos_mask_1].squeeze(-1)  # (num_pos_1,)
                    anc_xywhr_1 = xyxyxyxy2xywhr(
                        verts_1, ang_1, image_sizes[b]
                    )  # (num_pos_1, 5)

                # -------------------------------------
                # Stage 2: Match provisional OBBs to GT using IoU thresholds 2
                # -------------------------------------
                pos_mask_2, neg_mask_2, best_gt_2 = match_anchors_to_targets(
                    anc_xywhr_1,
                    targets["boxes"][b],
                    targets["angle"][b].squeeze(-1),
                    image_sizes[b],
                    pos_iou_thr=self.pos_iou_thr_2,
                    neg_iou_thr=self.neg_iou_thr_2,
                )
                # pos_mask_2 is shape (num_pos_1,) indicating which provisional are positive

                if pos_mask_2.any():
                    valid_batches += 1
                    # Map provisional indices back to absolute anchor indices
                    abs_pos_idx_2 = pos_mask_1.nonzero(as_tuple=False).squeeze(1)[
                        pos_mask_2
                    ]
                    # Retrieve ground-truth indices for stage 2 positives
                    gt_idx_2 = best_gt_2[pos_mask_2]  # (num_pos_2,)

                    # -------------------------------------
                    # Stage 2: Final OBB regression loss (only for stage-2 positives)
                    # -------------------------------------
                    pred_deltas_2 = deltas[b][abs_pos_idx_2]  # (num_pos_2, 8)
                    gt_boxes_2 = targets["boxes"][b][gt_idx_2]  # (num_pos_2, 8)
                    anc_xy_2 = anchors_xy[b][abs_pos_idx_2]  # (num_pos_2, 8)
                    obb_loss += self.obb_loss(
                        pred_deltas_2.unsqueeze(0),  # (1, num_pos_2, 8)
                        gt_boxes_2.unsqueeze(0),  # (1, num_pos_2, 8)
                        anc_xy_2.unsqueeze(0),  # (1, num_pos_2, 8)
                    )

                    # -------------------------------------
                    # Stage 2: Final rotation loss
                    # -------------------------------------
                    pa_2 = pred_angles[b][abs_pos_idx_2].squeeze(-1)  # (num_pos_2,)
                    ga_2 = targets["angle"][b][gt_idx_2].squeeze(-1)  # (num_pos_2,)
                    pa_wrapped_2 = wrap_to_pi(pa_2)  # (num_pos_2,)
                    ga_wrapped_2 = wrap_to_pi(ga_2)  # (num_pos_2,)
                    rot_loss += self.rot_loss(
                        pa_wrapped_2.unsqueeze(-1), ga_wrapped_2.unsqueeze(-1)
                    )

        # -------------------------------------
        # Normalize and combine all losses
        # -------------------------------------
        # Classification loss normalized by number of batches
        cls_loss = cls_loss / max(1, num_cls_batches)
        # Face loss normalized by batch size
        face_loss /= B
        if valid_batches > 0:
            obb_loss /= valid_batches
            rot_loss /= valid_batches
            total_loss = (
                self.lambda_cls * cls_loss
                + self.lambda_face * face_loss
                + self.lambda_obb * obb_loss
                + self.lambda_rot * rot_loss
            )
        else:
            # If no positives in stage 2 at all, fall back to only face loss
            total_loss = self.lambda_face * face_loss

        return (
            total_loss,
            cls_loss if not isinstance(cls_loss, torch.Tensor) else cls_loss.item(),
            face_loss if not isinstance(face_loss, torch.Tensor) else face_loss.item(),
            obb_loss if not isinstance(obb_loss, torch.Tensor) else obb_loss.item(),
            rot_loss if not isinstance(rot_loss, torch.Tensor) else rot_loss.item(),
        )
