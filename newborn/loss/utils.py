from typing import Tuple

import torch
import numpy as np
import math

from data_setup.augmentations import wrap_to_pi


def xyxyxyxy2xywhr(
    obb: torch.Tensor, angle: torch.Tensor, image_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Converts oriented bounding boxes from 8-point (x1, y1, ..., x4, y4) format plus angle
    to canonical (cx, cy, w, h, θ) format, robustly handling normalized or pixel coordinates.

    - If input vertices are normalized (max <= 1.0), scales them to pixel coordinates.
    - Reshapes input to (N, 4, 2) if needed.
    - Delegates conversion to 'verts_to_xywhr_with_theta' for canonicalization (width >= height, θ wrapped).

    Args:
        obb (torch.Tensor): Tensor of shape (N, 8) or (8,) containing the 4 vertices of each OBB.
        angle (torch.Tensor): Tensor of shape (N,) or (1,) with rotation angles in radians.
        image_size (Tuple[int, int]): (width, height) of the image for scaling.

    Returns:
        torch.Tensor: Tensor of shape (N, 5) in (cx, cy, w, h, θ) canonical format.
    """
    # Handle empty input
    if obb.numel() == 0:
        return obb.new_empty((0, 5))

    # Ensure input is (N, 8)
    if obb.ndim == 1:
        obb = obb.unsqueeze(0)
    N = obb.shape[0]
    device = obb.device
    W, H = image_size

    # If vertices are normalized (max <= 1.0), scale to pixel coordinates
    if obb.max() <= 1.0:
        scale = torch.tensor([W, H] * 4, device=device, dtype=obb.dtype)
        obb_pix = obb * scale
    else:
        obb_pix = obb

    # Reshape to (N, 4, 2) for vertex format
    verts = obb_pix.view(N, 4, 2)
    angle = angle.to(device).view(N)

    # Convert to canonical (cx, cy, w, h, θ) format
    return verts_to_xywhr_with_theta(verts, angle)


def get_covariance_matrix(boxes: torch.Tensor):
    """
    Computes the covariance components (a, b, c) of oriented bounding boxes (OBBs) in xywhr format.

    This representation is used for probabilistic IoU computation between Gaussian-modeled OBBs.
    The (x, y) center is ignored, and only width, height, and rotation are used.

    Args:
        boxes (torch.Tensor): Tensor of shape (N, 5) in (x, y, w, h, θ) format, where θ is in radians.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Covariance components a, b, c for each box.
    """
    # Convert width/height to variance form: (w²/12, h²/12)
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)  # a: var_x, b: var_y, c: rotation

    # Compute cos and sin of the rotation angle
    cos, sin = c.cos(), c.sin()
    cos2, sin2 = cos.pow(2), sin.pow(2)

    # Compute rotated covariance components
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def batch_probiou(
    obb1: torch.Tensor, obb2: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """
    Vectorized computation of probabilistic IoU between two sets of OBBs.

    Args:
        obb1 (torch.Tensor): Tensor of shape (N, 5) for GT boxes.
        obb2 (torch.Tensor): Tensor of shape (M, 5) for predicted boxes.
        eps (float): Small value for numerical stability.

    Returns:
        torch.Tensor: Tensor of shape (N, M) with probabilistic IoU values.
    """
    # Ensure input tensors are on the same device
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    # Extract x, y, width, height, and angle from the OBBs
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))

    # Compute covariance components for both sets of OBBs
    a1, b1, c1 = get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in get_covariance_matrix(obb2))

    # Mahalanobis-like distance between centers
    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
        * 0.25
    )

    # Cross-correlation term
    t2 = (
        ((c1 + c2) * (x2 - x1) * (y1 - y2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
        * 0.5
    )

    # Log-determinant divergence between covariances
    det1 = (a1 * b1 - c1.pow(2)).clamp(min=eps)
    det2 = (a2 * b2 - c2.pow(2)).clamp(min=eps)
    # Compute the log-determinant divergence
    # Note: det1 and det2 are positive, so we can safely take the square root
    # and add eps for numerical stability
    # Compute the log-determinant divergence
    num = (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps
    den = 4 * (det1 * det2).sqrt() + eps
    t3 = torch.log((num / den).clamp(min=eps)) * 0.5

    # Bhattacharyya distance (bounded)
    bd = (t1 + t2 + t3).clamp(min=eps, max=100.0)

    # Convert to similarity
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def match_anchors_to_targets(
    anchors_xywhr: torch.Tensor,  # (N, 5) — Precomputed anchor boxes in (x, y, w, h, θ)
    gt_boxes_xy: torch.Tensor,  # (M, 8) — Ground truth OBBs in 4-point (x1y1...x4y4) format
    gt_angles: torch.Tensor,  # (M,)   — Rotation angles in radians
    image_size: Tuple[int, int],  # (W, H)
    pos_iou_thr: float = 0.5,  # IoU threshold for positive match
    neg_iou_thr: float = 0.4,  # IoU threshold for negative match
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Matches precomputed anchors to ground truth oriented bounding boxes using pIoU.

    Converts the ground truth boxes from 8-point + angle format to xywhr,
    computes pairwise pIoU, and assigns the best matching GT to each anchor.

    Args:
        anchors_xywhr (torch.Tensor): Anchor boxes, shape (N, 5), format (x, y, w, h, θ).
        gt_boxes_xy (torch.Tensor): Ground truth boxes in xyxyxyxy format, shape (M, 8).
        gt_angles (torch.Tensor): Rotation angles for GT boxes in radians, shape (M,).
        image_size (Tuple[int, int]): Size of the image (width, height).
        pos_iou_thr (float): IoU threshold for positive match.
        neg_iou_thr (float): IoU threshold for negative match.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - pos_mask (torch.Tensor): Positive matches, shape (N,).
            - neg_mask (torch.Tensor): Negative matches, shape (N,).
            - best_gt (torch.Tensor): Indices of the best matching GT for each anchor, shape (N,).
    """
    # Ensure input tensors are on the same device
    N = anchors_xywhr.size(0)

    # Check if there are no ground truth boxes
    if gt_boxes_xy.numel() == 0:
        # If no GT boxes, return empty masks
        return (
            torch.zeros(N, dtype=torch.bool, device=anchors_xywhr.device),  # pos
            torch.zeros(N, dtype=torch.bool, device=anchors_xywhr.device),  # neg
            torch.full(
                (N,), -1, dtype=torch.long, device=anchors_xywhr.device
            ),  # best_gt
        )

    # Get the image size
    W, H = image_size
    # Convert anchors to xywhr format
    gt_xywhr = xyxyxyxy2xywhr(gt_boxes_xy, gt_angles, (W, H))

    # Compute pairwise pIoU between anchors and GT boxes
    iou_matrix = batch_probiou(anchors_xywhr, gt_xywhr)
    # Find the best matching GT for each anchor
    best_iou, best_gt = iou_matrix.max(dim=1)
    # Find the best matching anchor for each GT
    pos_mask = best_iou > pos_iou_thr  # Positive if IoU is greater than threshold
    neg_mask = best_iou <= neg_iou_thr  # Negative if IoU is less than threshold

    # Return masks and best GT indices
    return pos_mask, neg_mask, best_gt


def probiou(
    obb1: torch.Tensor, obb2: torch.Tensor, CIoU: bool = False, eps: float = 1e-7
) -> torch.Tensor:
    """
    Computes the probabilistic IoU (pIoU) between oriented bounding boxes.

    Optionally includes the Complete IoU (CIoU) term, penalizing aspect ratio mismatches,
    inspired by standard CIoU in axis-aligned box regression.

    Args:
        obb1 (torch.Tensor): Ground truth OBBs of shape (N, 5), format (x, y, w, h, θ).
        obb2 (torch.Tensor): Predicted OBBs of shape (N, 5), format (x, y, w, h, θ).
        CIoU (bool, optional): If True, adds the CIoU penalty term. Default is False.
        eps (float, optional): Small constant for numerical stability. Default is 1e-7.

    Returns:
        torch.Tensor: Probabilistic IoU values for each pair, shape (N,).

    References:
        - https://arxiv.org/pdf/2106.06072v1.pdf (pIoU for Gaussian Boxes)
        - https://arxiv.org/abs/1911.08287 (CIoU term inspiration)
    """
    # Ensure input tensors are on the same device
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)

    # Compute covariance components for both sets of OBBs
    a1, b1, c1 = get_covariance_matrix(obb1)
    a2, b2, c2 = get_covariance_matrix(obb2)

    # Mahalanobis-like center distance
    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
        * 0.25
    )

    # Cross-correlation term
    t2 = (
        ((c1 + c2) * (x2 - x1) * (y1 - y2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
        * 0.5
    )

    # Covariance divergence
    det1 = (a1 * b1 - c1.pow(2)).clamp(min=eps)
    det2 = (a2 * b2 - c2.pow(2)).clamp(min=eps)
    # Compute the log-determinant divergence
    # Note: det1 and det2 are positive, so we can safely take the square root
    # and add eps for numerical stability
    # Compute the log-determinant divergence
    num = (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps
    den = 4 * (det1 * det2).sqrt() + eps
    t3 = torch.log((num / den).clamp(min=eps)) * 0.5

    # Bhattacharyya distance
    bd = (t1 + t2 + t3).clamp(min=eps, max=100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd

    # Optional CIoU penalty
    if CIoU:
        # Compute aspect ratio penalty
        # Convert width/height to variance form: (w²/12, h²/12)

        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            # Compute the aspect ratio penalty
            alpha = v / (v - iou + (1 + eps))
        return iou - alpha * v  # CIoU

    return iou


def decode_vertices(
    deltas: torch.Tensor,  # (N, 8) — Predicted normalized offsets from anchor vertices
    anchors: torch.Tensor,  # (N, 8) — Anchor box vertices in pixel coordinates
    pred_angles: torch.Tensor,  # (N,) — Predicted box rotation angles in radians
    image_size: Tuple[int, int],  # (W, H)
    scale: float = 1.0,  # Scale factor for offsets
    clamp: bool = False,  # Whether to clamp output vertices to image bounds
) -> torch.Tensor:
    """
    Decodes predicted normalized vertex offsets into absolute OBB vertex coordinates.

    This function reconstructs the absolute positions of the 4 vertices of an oriented bounding box (OBB)
    from predicted normalized offsets (deltas) relative to anchor box vertices. The deltas are first
    scaled by the anchor's diagonal length (for scale invariance), then rotated by the predicted angle,
    and finally added to the anchor vertices to obtain the decoded absolute positions in pixel coordinates.
    The resulting vertices are clamped to the image bounds.

    Args:
        deltas (torch.Tensor): Tensor of shape (N, 8) containing predicted normalized offsets for each
            of the 4 vertices (x, y) of the OBB, typically in [-1, 1].
        anchors (torch.Tensor): Tensor of shape (N, 8) with anchor box vertex coordinates in pixel space,
            ordered as (x0, y0, x1, y1, x2, y2, x3, y3).
        pred_angles (torch.Tensor): Tensor of shape (N,) with predicted rotation angles in radians.
        image_size (Tuple[int, int]): Tuple (width, height) specifying the image dimensions for clamping.
        scale (float, optional): Scaling factor applied to the offset magnitude. Default is 1.0.
        clamp (bool, optional): If True, clamps the output vertex coordinates to the image bounds. Default is False.

    Returns:
        torch.Tensor: Tensor of shape (N, 8) containing the decoded absolute vertex positions for each OBB,
            clamped to the image bounds.
    """
    # Get the image dimensions and number of anchors
    W, H = image_size
    # Get the number of anchors
    N = anchors.size(0)
    # Reshape angles and anchors to (N, 4, 2) for easier manipulation
    pred_angles = pred_angles.reshape(N)
    anc_xy = anchors.view(N, 4, 2)

    # Scale normalized offsets by anchor diagonal and scale factor
    diag = (anc_xy[:, 0] - anc_xy[:, 2]).norm(dim=1, keepdim=True) * scale
    offs = deltas.view(N, 4, 2) * diag.unsqueeze(-1)

    # Rotate offsets by predicted angle for each box
    cos, sin = pred_angles.cos().unsqueeze(1), pred_angles.sin().unsqueeze(1)
    dx, dy = offs[..., 0], offs[..., 1]
    # Apply rotation: (cos, -sin) and (sin, cos) for each box
    rot_x = cos * dx - sin * dy
    rot_y = sin * dx + cos * dy
    offs_rot = torch.stack((rot_x, rot_y), dim=-1)

    # Add rotated offsets to anchor vertices to get decoded vertices
    verts = anc_xy + offs_rot  # (N, 4, 2)

    # Clamp coordinates to image bounds
    if clamp:
        verts[..., 0].clamp_(0, W)
        verts[..., 1].clamp_(0, H)
    else:
        verts[..., 0].clamp_(-W, W * 2) # Allow some leeway for boxes slightly outside the image
        verts[..., 1].clamp_(-H, H * 2) 
    return verts.view(N, 8)


def xywhr2xyxyxyxy(xywhr: torch.Tensor) -> np.ndarray:
    """
    Converts a batch of boxes from xywhr format to 4 corner coordinates (N, 4, 2).
    The xywhr format is defined as (cx, cy, w, h, angle), where:
        - cx, cy: center coordinates
        - w, h: width and height
        - angle: rotation angle in radians
    The output format is a list of 4 corner points for each box in the order:
        [top-left, top-right, bottom-right, bottom-left].
    Args:
        xywhr (torch.Tensor): Tensor of shape (N, 5) in xywhr format.
    Returns:
        np.ndarray: Array of shape (N, 4, 2) with corner coordinates.
    """
    # If you get a single box tensor of shape (5,), make it (1,5)
    if xywhr.ndim == 1:
        xywhr = xywhr.unsqueeze(0)

    # Now xywhr is (N,5); unpack along dim=1
    cx, cy, w, h, angle = xywhr.unbind(dim=1)  # each is shape (N,)

    # Half-dimensions
    dx = w * 0.5  # (N,)
    dy = h * 0.5  # (N,)

    # Build the 4 corners relative to center before rotation:
    # shape (N,4,2)
    offsets = torch.stack(
        [
            torch.stack([-dx, -dy], dim=1),  # top-left
            torch.stack([dx, -dy], dim=1),  # top-right
            torch.stack([dx, dy], dim=1),  # bottom-right
            torch.stack([-dx, dy], dim=1),  # bottom-left
        ],
        dim=1,
    )

    # Rotation matrices for each box: shape (N,2,2)
    cos_a = angle.cos()
    sin_a = angle.sin()
    rot_mats = torch.stack(
        [
            torch.stack([cos_a, -sin_a], dim=1),
            torch.stack([sin_a, cos_a], dim=1),
        ],
        dim=1,
    )

    # Apply rotation: (N,4,2) @ (N,2,2) -> (N,4,2)
    rotated = torch.bmm(offsets, rot_mats)

    # Translate to the true centers
    centers = torch.stack([cx, cy], dim=1).unsqueeze(1)  # (N,1,2)
    corners_abs = rotated + centers  # (N,4,2)

    return corners_abs.detach().cpu().numpy()


def encode_vertices(
    gt_boxes: torch.Tensor,  # (N, 8) absolute vertex coordinates of ground truth OBBs
    anchors: torch.Tensor,  # (N, 8) absolute vertex coordinates of anchor OBBs
    gt_angles: torch.Tensor,  # (N,) rotation angles of ground truth OBBs in radians
    scale: float = 1.0,  # Scale factor for normalization
) -> torch.Tensor:
    """
    Encodes ground truth oriented bounding boxes (OBBs) as normalized deltas
    relative to anchor boxes, for use as regression targets.

    For each OBB, computes the offset between each ground truth vertex and the
    corresponding anchor vertex, rotates this offset by the inverse of the ground
    truth angle (to align with the canonical anchor orientation), and normalizes
    by the anchor's diagonal length (between vertex 0 and 2) times the scale factor.

    The resulting deltas are clamped to [-1, 1] and can be used as regression targets
    for OBB vertex prediction.

    Args:
        gt_boxes (torch.Tensor): Ground truth boxes in absolute vertex format (N, 8).
        anchors (torch.Tensor): Anchor boxes in absolute vertex format (N, 8).
        gt_angles (torch.Tensor): Rotation angles of ground truth OBBs in radians (N,).
        scale (float, optional): Scale factor for normalization. Default is 0.5.

    Returns:
        torch.Tensor: Normalized deltas of shape (N, 8), where each value is in Δx/Δy space,
                      clamped to [-1, 1].
    """
    if gt_angles.dim() > 1:  # (N,1) -> (N,)
        gt_angles = gt_angles.squeeze(-1)

    N = anchors.size(0)
    anc_xy = anchors.view(N, 4, 2)
    gt_xy = gt_boxes.view(N, 4, 2)

    # Compute offsets between ground truth and anchor vertices
    dx = gt_xy[..., 0] - anc_xy[..., 0]
    dy = gt_xy[..., 1] - anc_xy[..., 1]

    # Compute inverse rotation matrices for each box (by -gt_angle)
    cos_t = gt_angles.cos().unsqueeze(1)  # (N,1)
    sin_t = gt_angles.sin().unsqueeze(1)  # (N,1)

    # Apply inverse rotation (-θ) to offsets
    rot_x = cos_t * dx + sin_t * dy  # (N,4)
    rot_y = -sin_t * dx + cos_t * dy  # (N,4)

    offs = torch.stack([rot_x, rot_y], dim=-1)  # (N,4,2)

    # Normalize by the anchor diagonal length and scale factor
    diag = (anc_xy[:, 0] - anc_xy[:, 2]).norm(dim=1, keepdim=True)  # (N,1)
    offs = offs / (diag.unsqueeze(-1) * scale)  # (N,4,2)

    return offs.reshape(N, 8)

def verts_to_xywhr_with_theta(verts: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Converts vertices of oriented bounding boxes to canonical (x, y, w, h, θ) format.

    Takes a set of 4 vertices defining oriented bounding boxes and their rotation angles,
    and converts them to center coordinates, width, height and canonical rotation angle.
    The rotation is wrapped to [-π, π] and width/height are ordered so width >= height.

    Args:
        verts (torch.Tensor): Tensor of shape (N, 4, 2) or (N, 8) containing vertices
                             of oriented bounding boxes.
        theta (torch.Tensor): Tensor of shape (N,) containing rotation angles in radians.

    Returns:
        torch.Tensor: Tensor of shape (N, 5) in (cx, cy, w, h, θ) format where:
            - cx, cy: center coordinates
            - w, h: width and height (w >= h)
            - θ: rotation angle in radians, wrapped to [-π, π]
    """
    # Ensure vertices are in (N,4,2) format
    if verts.ndim == 2:
        verts = verts.view(-1, 4, 2)
    N = verts.size(0)
    theta = theta.view(N)

    # Get box centers by averaging vertices
    c = verts.mean(dim=1, keepdim=True)  # (N,1,2)
    rel = verts - c  # Center vertices at origin

    # Create unit vectors along rotated axes
    u = torch.stack([theta.cos(), theta.sin()], dim=1).unsqueeze(1)  # Primary axis
    v = torch.stack([-theta.sin(), theta.cos()], dim=1).unsqueeze(
        1
    )  # Perpendicular axis

    # Project vertices onto rotated axes
    x = (rel * u).sum(dim=-1)  # Projections on primary axis
    y = (rel * v).sum(dim=-1)  # Projections on perpendicular axis

    # Compute width and height as span of projections
    w = x.max(1).values - x.min(1).values
    h = y.max(1).values - y.min(1).values

    # Extract center coordinates and wrap angle to [-π, π]
    cx = c[..., 0].squeeze(1)
    cy = c[..., 1].squeeze(1)
    th = wrap_to_pi(theta)

    # Return in canonical format
    return torch.stack([cx, cy, w, h, th], dim=1)
