from typing import List, Tuple, Dict, Union, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    match_anchors_to_targets,
    probiou,
    xyxyxyxy2xywhr,
    decode_vertices,
    encode_vertices,
    verts_to_xywhr_with_theta,
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
        ignore_index: int = -1,
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
    """
    Least Squares (L2/MSE) Loss for classification with optional Gaussian label smoothing.

    Theory:
        The classic L2 loss for classification is:
            L = 0.5 * sum_c α_c * (p_c - y_c)^2
        where:
            - p_c: predicted probability for class c (after softmax)
            - y_c: target probability for class c (one-hot or soft label)
            - α_c: class balancing weight (optional)

        If sigma is None, y_c is a one-hot vector (classic MSE for classification).
        If sigma > 0, y_c is a soft label vector with a Gaussian distribution centered at the target class,
        providing label smoothing and encouraging the model to distribute probability mass around the true class.

    Args:
        alpha (float or list[float]): Class balancing weights (α_c). If float, same for all classes.
        sigma (float, optional): Standard deviation for Gaussian label smoothing. If None, uses one-hot labels.
        ignore_index (int): Label value to ignore in loss computation.
        reduction (str): 'mean', 'sum', or 'none' for loss reduction.
    """

    def __init__(
        self,
        alpha: Union[float, List[float]] = 1.0,
        sigma: Optional[float] = None,
        ignore_index: int = -1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.sigma = sigma
        # Convert alpha to tensor for broadcasting
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = torch.tensor([alpha], dtype=torch.float32)
        self.ignore_index = ignore_index
        self.reduction = reduction

    @staticmethod
    def labels_gaussian_soft(idx: torch.Tensor, C: int, sigma: float) -> torch.Tensor:
        """
        Generate soft labels using a Gaussian distribution centered at the target class.

        Args:
            idx (Tensor): Target class indices, shape (N,)
            C (int): Number of classes
            sigma (float): Standard deviation for Gaussian smoothing

        Returns:
            Tensor: Soft label matrix of shape (N, C)
        """
        grid = torch.arange(
            C, device=idx.device
        ).float()  # Class indices [0, 1, ..., C-1]
        idx = idx.unsqueeze(1).float()  # Shape (N, 1) for broadcasting
        dist2 = (grid - idx).pow(2)  # Squared distance to target class
        probs = torch.exp(-dist2 / (2 * sigma**2))  # Gaussian weights for each class
        return probs / probs.sum(dim=1, keepdim=True)  # Normalize to sum to 1

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the L2/MSE classification loss.

        Args:
            logits (Tensor): Raw model outputs, shape (..., C)
            targets (Tensor): Target class indices, shape (...)

        Returns:
            Tensor: Scalar loss (if reduction), or per-sample loss
        """
        # Number of classes
        C = logits.shape[-1]

        # Predicted probabilities, shape (..., C)
        p = F.softmax(logits, dim=-1)

        # Flatten targets to (N,)
        t = targets.view(-1)

        # Mask for valid targets
        mask = t != self.ignore_index

        # Return zero if no valid targets
        if mask.sum() == 0:
            return logits.new_tensor(0.0)

        # Select valid predictions, shape (M, C)
        p = p.view(-1, C)[mask]

        # Select valid targets, shape (M,)
        t = t[mask]

        # Use one-hot labels if no smoothing
        if self.sigma is None:
            y = F.one_hot(t, C).float()
        # Use Gaussian soft labels
        else:
            y = self.labels_gaussian_soft(t, C, self.sigma)

        # Prepare alpha for broadcasting: shape (C,)
        alpha = (
            self.alpha.to(p.device)
            if self.alpha.numel() == C
            else self.alpha.to(p.device).expand(C)
        )

        # Compute weighted L2 loss for each sample: 0.5 * sum_c α_c * (p_c - y_c)^2
        loss = 0.5 * ((p - y).pow(2) * alpha).sum(dim=1)

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            # No reduction, return per-sample loss
            return loss


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
        Oriented Bounding Box (OBB) regression loss for 8-point vertex parameterization.

        This loss computes the distance between predicted and ground-truth OBBs in the vertex space.
        The predicted deltas are compared to the encoded ground-truth deltas (relative to the anchor and GT angle).

        Supports two loss types:
            - "l1": Standard L1 loss (mean absolu
    te error) between predicted and ground-truth deltas.
            - "smooth_l1": Smooth L1 loss (Huber loss) for robustness to outliers.

        Args:
            loss_type (str): "l1" for L1 loss, "smooth_l1" for Smooth-L1 loss.
            beta (float): Transition point for Smooth-L1 loss. Ignored if loss_type="l1".
            reduction (str): Reduction method: "none", "sum", or "mean".
    """

    def __init__(
        self,
        loss_type: str = "l1",
        beta: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            loss_type (str): "l1" for L1 loss, "smooth_l1" for Smooth-L1 loss.
            beta (float): Transition point for Smooth-L1 loss. Ignored if loss_type="l1".
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
        gt_angles: torch.Tensor,  # (B=1, N_pos, 1) or (N_pos, 1)
        gt_xy: torch.Tensor,  # (B=1, N_pos, 8) or (N_pos, 8)
        anchors: torch.Tensor,  # (B=1, N_pos, 8) or (N_pos, 8)
    ) -> torch.Tensor:
        """
        Computes the OBB regression loss between predicted and ground-truth deltas.

        Args:
            pred_deltas (Tensor): Predicted vertex deltas, shape (B=1, N, 8) or (N, 8).
            gt_angles (Tensor): Ground-truth angles in radians, shape (B=1, N, 1) or (N, 1).
            gt_xy (Tensor): Ground-truth vertices, shape (B=1, N, 8) or (N, 8).
            anchors (Tensor): Anchor box vertices, shape (B=1, N, 8) or (N, 8).

        Returns:
            Tensor: Scalar loss value (if reduction is "mean" or "sum"), or per-sample loss.
        """
        # Remove leading batch dimension if present (B=1)
        if pred_deltas.dim() == 3 and pred_deltas.size(0) == 1:
            pred_deltas = pred_deltas.squeeze(0)
            gt_angles = gt_angles.squeeze(0)
            gt_xy = gt_xy.squeeze(0)
            anchors = anchors.squeeze(0)

        if gt_angles.dim() == 2 and gt_angles.size(1) == 1:
            gt_angles = gt_angles.squeeze(1)

        # Encode ground-truth vertices as deltas relative to anchors and GT angle
        gt_deltas = encode_vertices(gt_xy, anchors, gt_angles)

        # Compute the selected loss between predicted and ground-truth deltas
        if self.loss_type == "l1":
            return F.l1_loss(pred_deltas, gt_deltas, reduction=self.reduction)
        else:  # smooth_l1
            return F.smooth_l1_loss(
                pred_deltas, gt_deltas, beta=self.beta, reduction=self.reduction
            )


def orthogonality_loss(verts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the orthogonality loss for a set of vertices representing a rectangle.

    The loss encourages the following properties:
        1. Orthogonality between adjacent edges (should be close to 90 degrees).
        2. Parallelism between opposite edges (should be close to parallel).
        3. Equality of lengths of opposite edges (invariant to global scale).

    Args:
        verts (torch.Tensor): Tensor of shape (N, 8) or (N, 4, 2) representing the vertices of rectangles.
                              Each rectangle is defined by its vertices in either vertex format or as pairs of coordinates.
        eps (float): Small value to prevent division by zero in calculations.

    Returns:
        torch.Tensor: Mean orthogonality loss across all rectangles.
    """
    # Reshape vertices to (N, 4, 2) if they are in (N, 8) format
    if verts.ndim == 2:
        verts = verts.view(-1, 4, 2)

    # Calculate edge vectors
    e0 = verts[:, 1] - verts[:, 0]  # Edge from v0 to v1
    e1 = verts[:, 2] - verts[:, 1]  # Edge from v1 to v2
    e2 = verts[:, 3] - verts[:, 2]  # Edge from v2 to v3
    e3 = verts[:, 0] - verts[:, 3]  # Edge from v3 to v0

    def cos2(a, b):
        """Calculate the squared cosine of the angle between two vectors."""
        num = (a * b).sum(-1)
        den = (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(eps)
        return (num / den).pow(2)

    # (1) Orthogonality between adjacent edges: cos^2 should be close to 0
    L_orth = cos2(e0, e1) + cos2(e1, e2) + cos2(e2, e3) + cos2(e3, e0)

    # # (2) Parallelism between opposite edges: |cos| should be close to 1
    # par02 = (e0 * e2).sum(-1) / (e0.norm(dim=-1) * e2.norm(dim=-1) + eps)
    # par13 = (e1 * e3).sum(-1) / (e1.norm(dim=-1) * e3.norm(dim=-1) + eps)
    # L_par = (1 - par02.abs()) + (1 - par13.abs())

    # # (3) Equality of lengths of opposite edges
    # def len_term(a, b):
    #     la, lb = a.norm(dim=-1), b.norm(dim=-1)
    #     return ((la - lb) / (la + lb + eps)).pow(2)

    # L_len = len_term(e0, e2) + len_term(e1, e3)

    # Return the mean loss across all rectangles
    return (L_orth).mean()


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with optional homoscedastic-uncertainty weighting (Kendall et al., CVPR 2018).

    If `use_uncertainty=True`, each task i has a learnable log-variance parameter s_i = log(sigma_i^2),
    and the combined objective is:

        L_total = sum_{i in enabled} exp(-s_i) * L_i + s_i

    Important:
        - The +s_i term must only be added when the task loss L_i is computed (enabled),
          otherwise s_i can be driven to -inf to reduce the loss without supervision.
        - It is usually safer to keep pure regularizers (e.g., orthogonality) with fixed weights.
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
        lambda_rect: float = config.LAMBDA_RECT,
        lambda_child: float = config.LAMBDA_CHILD,
        pos_iou_thr_1: float = config.POS_IOU_THRESH_1,
        neg_iou_thr_1: float = config.NEG_IOU_THRESH_1,
        pos_iou_thr_2: float = config.POS_IOU_THRESH_2,
        neg_iou_thr_2: float = config.NEG_IOU_THRESH_2,
        alpha: List[float] = config.ALPHA,
        gamma: float = config.GAMMA,
        neg_samples_ratio: int = config.NEG_SAMPLES_RATIO,
        face_pos_weight: float = config.FACE_POS_WEIGHT,
        sigma_l2_cls: float = config.SIGMA_L2_CLS,
        use_uncertainty: bool = True,
        include_rect_in_uncertainty: bool = False,
        log_var_clamp: Optional[Tuple[float, float]] = (-10.0, 10.0),
    ) -> None:
        super().__init__()

        if cls_loss_type == "focal":
            self.cls_loss_fn = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
        elif cls_loss_type == "ls":
            self.cls_loss_fn = L2Loss(alpha=alpha, reduction="mean", sigma=sigma_l2_cls)
        else:
            raise ValueError("cls_loss_type must be 'focal' or 'ls'")

        self.face_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(face_pos_weight), reduction="mean"
        )
        self.child_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.obb_loss_fn = OBBRegressionLoss(
            loss_type=obb_loss_type, beta=2.0, reduction="mean"
        )
        self.rot_loss_fn = RotationLoss(mode=rot_loss_type)

        self.lambda_rect = lambda_rect  # keep fixed by default

        self.pos_iou_thr_1 = pos_iou_thr_1
        self.neg_iou_thr_1 = neg_iou_thr_1
        self.pos_iou_thr_2 = pos_iou_thr_2
        self.neg_iou_thr_2 = neg_iou_thr_2
        self.neg_samples_ratio = neg_samples_ratio

        self.use_uncertainty = use_uncertainty
        self.include_rect_in_uncertainty = include_rect_in_uncertainty
        self.log_var_clamp = log_var_clamp

        # Initialize learnable log-variances so that exp(-s) roughly matches your old lambdas.
        # If weight w_i = lambda_i, then choose s_i = -log(w_i).
        def init_log_var(weight: float) -> float:
            w = max(float(weight), 1e-8)
            return -math.log(w)

        self.log_vars = nn.ParameterDict(
            {
                "cls": nn.Parameter(
                    torch.tensor(init_log_var(lambda_cls), dtype=torch.float32)
                ),
                "face": nn.Parameter(
                    torch.tensor(init_log_var(lambda_face), dtype=torch.float32)
                ),
                "child": nn.Parameter(
                    torch.tensor(init_log_var(lambda_child), dtype=torch.float32)
                ),
                "obb": nn.Parameter(
                    torch.tensor(init_log_var(lambda_obb), dtype=torch.float32)
                ),
                "rot": nn.Parameter(
                    torch.tensor(init_log_var(lambda_rot), dtype=torch.float32)
                ),
            }
        )

        if self.include_rect_in_uncertainty:
            self.log_vars["rect"] = nn.Parameter(
                torch.tensor(init_log_var(lambda_rect), dtype=torch.float32)
            )

    def _weighted_task_term(
        self, loss_value: torch.Tensor, log_var: torch.Tensor, enabled: bool
    ) -> torch.Tensor:
        """
        Compute exp(-s) * L + s for a single task, gated by enabled flag.

        Args:
            loss_value: Scalar tensor loss for the task.
            log_var: Learnable scalar tensor s = log(sigma^2).
            enabled: Whether this task loss was computed on this forward pass.

        Returns:
            Scalar tensor contribution to total loss.
        """
        if not enabled:
            return loss_value.new_tensor(0.0)

        s = log_var
        if self.log_var_clamp is not None:
            s = torch.clamp(s, self.log_var_clamp[0], self.log_var_clamp[1])

        weight = torch.exp(-s)
        return weight * loss_value + 0.5 * s

    def forward(
        self,
        preds: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        targets: Dict[str, torch.Tensor],
        anchors_xy: torch.Tensor,
        anchors_xywhr: torch.Tensor,
        image_sizes: List[Tuple[int, int]],
    ):
        """
        Compute multi-task loss.

        Returns:
            (total_loss, cls, face, obb, rot, rect, child) where components are Python floats.
        """
        orient_logits, face_logits, deltas, pred_angles, child_logits = preds
        B, N, _ = orient_logits.shape
        device = orient_logits.device

        cls_loss = torch.tensor(0.0, device=device)
        face_loss = torch.tensor(0.0, device=device)
        obb_loss = torch.tensor(0.0, device=device)
        rot_loss = torch.tensor(0.0, device=device)
        rect_loss = torch.tensor(0.0, device=device)
        child_loss = torch.tensor(0.0, device=device)

        face_batches = 0
        child_batches = 0
        cls_batches = 0
        stage2_batches = 0

        pred_dtype = orient_logits.dtype
        anchors_xy = anchors_xy.to(dtype=torch.float32)
        anchors_xywhr = anchors_xywhr.to(dtype=torch.float32)

        for b in range(B):
            pos_mask_1, neg_mask_1, best_gt_1 = match_anchors_to_targets(
                anchors_xywhr,
                targets["boxes"][b],
                targets["angle"][b].squeeze(-1),
                image_sizes[b],
                pos_iou_thr=self.pos_iou_thr_1,
                neg_iou_thr=self.neg_iou_thr_1,
            )
            pos_idx_1 = pos_mask_1.nonzero(as_tuple=False).squeeze(1)
            neg_idx_1 = neg_mask_1.nonzero(as_tuple=False).squeeze(1)

            if pos_idx_1.numel() == 0:
                continue

            face_batches += 1

            # Face loss with hard negative mining
            neg_logits_all = face_logits[b][neg_idx_1]
            with torch.no_grad():
                per_neg_loss = F.binary_cross_entropy_with_logits(
                    neg_logits_all, torch.zeros_like(neg_logits_all), reduction="none"
                ).view(-1)
            _, hard_order = per_neg_loss.sort(descending=True)
            num_hard = max(
                1, min(hard_order.numel(), pos_idx_1.numel() * self.neg_samples_ratio)
            )
            hard_neg_idx = neg_idx_1[hard_order[:num_hard]]
            sel_idx_1 = torch.cat([pos_idx_1, hard_neg_idx], dim=0)
            tgt_face = pos_mask_1.float().unsqueeze(1)[sel_idx_1]
            face_loss = face_loss + self.face_loss_fn(
                face_logits[b][sel_idx_1], tgt_face
            )

            # Child loss (only valid matched positives)
            valid_child_mask = best_gt_1[pos_idx_1] != -1
            if valid_child_mask.any():
                pos_idx_1_valid = pos_idx_1[valid_child_mask]
                tgt_child = targets["child_prob"][b][best_gt_1[pos_idx_1_valid]].to(
                    device
                )
                child_loss = child_loss + self.child_loss_fn(
                    child_logits[b][pos_idx_1_valid], tgt_child
                )
                child_batches += 1
                baby_mask = tgt_child.squeeze(1).bool()
            else:
                baby_mask = torch.zeros(0, dtype=torch.bool, device=device)

            # Orientation classification (only baby anchors)
            if baby_mask.numel() and baby_mask.any():
                pos_idx_1_baby = pos_idx_1_valid[baby_mask]
                valid_cls_mask = best_gt_1[pos_idx_1_baby] != -1
                if valid_cls_mask.any():
                    pos_idx_1_baby_valid = pos_idx_1_baby[valid_cls_mask]
                    tgt_cls_baby = targets["class_idx"][b][
                        best_gt_1[pos_idx_1_baby_valid]
                    ]
                    cls_loss = cls_loss + self.cls_loss_fn(
                        orient_logits[b][pos_idx_1_baby_valid], tgt_cls_baby
                    )
                    cls_batches += 1

            # Stage 2 requires baby anchors
            if not (baby_mask.numel() and baby_mask.any()):
                continue

            pos_mask_1_baby = torch.zeros_like(pos_mask_1)
            pos_mask_1_baby[pos_idx_1[baby_mask]] = True

            with torch.no_grad():
                pred_deltas_1 = deltas[b][pos_mask_1_baby]
                anc_xy_1 = anchors_xy[pos_mask_1_baby]
                ang_1 = pred_angles[b][pos_mask_1_baby].squeeze(-1)
                verts_1 = decode_vertices(
                    pred_deltas_1, anc_xy_1, ang_1, image_sizes[b]
                )
                anc_xywhr_1 = verts_to_xywhr_with_theta(verts_1, ang_1)

            pos_mask_2, _, best_gt_2 = match_anchors_to_targets(
                anc_xywhr_1,
                targets["boxes"][b],
                targets["angle"][b].squeeze(-1),
                image_sizes[b],
                pos_iou_thr=self.pos_iou_thr_2,
                neg_iou_thr=self.neg_iou_thr_2,
            )
            valid_gt_mask_2 = best_gt_2[pos_mask_2] != -1
            if not valid_gt_mask_2.any():
                continue

            abs_pos_idx_2 = pos_mask_1_baby.nonzero(as_tuple=False).squeeze(1)[
                pos_mask_2
            ][valid_gt_mask_2]
            gt_idx_2 = best_gt_2[pos_mask_2][valid_gt_mask_2]

            pred_deltas_2 = deltas[b][abs_pos_idx_2]
            gt_boxes_2 = targets["boxes"][b][gt_idx_2]
            anc_xy_2 = anchors_xy[abs_pos_idx_2]
            gt_angle_2 = wrap_to_pi(targets["angle"][b][gt_idx_2].squeeze(-1))

            obb_canonical = self.obb_loss_fn(
                pred_deltas_2.unsqueeze(0),
                gt_angle_2.unsqueeze(0).unsqueeze(-1),
                gt_boxes_2.unsqueeze(0),
                anc_xy_2.unsqueeze(0),
            )

            pred_angle_2 = pred_angles[b][abs_pos_idx_2].squeeze(-1)
            rot_loss = rot_loss + self.rot_loss_fn(
                pred_angle_2.unsqueeze(-1), gt_angle_2.unsqueeze(-1)
            )

            verts_pred = decode_vertices(
                pred_deltas_2, anc_xy_2, pred_angle_2, image_sizes[b]
            ).view(-1, 4, 2)
            rect_loss = rect_loss + orthogonality_loss(verts_pred)

            recon_loss = F.smooth_l1_loss(
                verts_pred.view(-1, 8),
                gt_boxes_2.view(-1, 8),
                reduction="mean",
            )

            obb_loss = obb_loss + 0.5 * obb_canonical + 0.5 * recon_loss
            stage2_batches += 1

        # Normalize (keep your logic)
        if cls_batches:
            cls_loss = cls_loss / cls_batches
        if face_batches:
            face_loss = face_loss / face_batches
        if child_batches:
            child_loss = child_loss / child_batches
        if stage2_batches:
            obb_loss = obb_loss / stage2_batches
            rot_loss = rot_loss / stage2_batches
            rect_loss = rect_loss / stage2_batches

        # Enable flags prevent +s_i when no supervision happened
        cls_enabled = cls_batches > 0
        face_enabled = face_batches > 0
        child_enabled = child_batches > 0
        stage2_enabled = stage2_batches > 0

        if self.use_uncertainty:
            total_loss = (
                self._weighted_task_term(cls_loss, self.log_vars["cls"], cls_enabled)
                + self._weighted_task_term(
                    face_loss, self.log_vars["face"], face_enabled
                )
                + self._weighted_task_term(
                    child_loss, self.log_vars["child"], child_enabled
                )
                + self._weighted_task_term(
                    obb_loss, self.log_vars["obb"], stage2_enabled
                )
                + self._weighted_task_term(
                    rot_loss, self.log_vars["rot"], stage2_enabled
                )
            )

            if self.include_rect_in_uncertainty:
                total_loss = total_loss + self._weighted_task_term(
                    rect_loss, self.log_vars["rect"], stage2_enabled
                )
            else:
                total_loss = total_loss + self.lambda_rect * rect_loss
        else:
            # Fallback to your fixed lambdas, if you want to compare.
            total_loss = (
                config.LAMBDA_CLS * cls_loss
                + config.LAMBDA_FACE * face_loss
                + config.LAMBDA_CHILD * child_loss
                + config.LAMBDA_OBB * obb_loss
                + config.LAMBDA_ROT * rot_loss
                + self.lambda_rect * rect_loss
            )

        return (
            total_loss,
            float(cls_loss.detach().item()),
            float(face_loss.detach().item()),
            float(obb_loss.detach().item()),
            float(rot_loss.detach().item()),
            float(rect_loss.detach().item()),
            float(child_loss.detach().item()),
        )

    @torch.no_grad()
    def get_task_weights(self) -> Dict[str, float]:
        """
        Return the current effective weights for each task: w_i = exp(-log_var_i).

        Returns:
            Dictionary mapping task name -> scalar float weight.
        """
        if not hasattr(self, "log_vars"):
            return {}

        weights: Dict[str, float] = {}
        for name, log_var in self.log_vars.items():
            weights[name] = float(torch.exp(-log_var.detach()).cpu().item())
        return weights

    @torch.no_grad()
    def get_task_log_vars(self) -> Dict[str, float]:
        """
        Return current log-variances (s_i) for each task.

        Returns:
            Dictionary mapping task name -> scalar float log variance.
        """
        if not hasattr(self, "log_vars"):
            return {}

        return {
            name: float(v.detach().cpu().item()) for name, v in self.log_vars.items()
        }
