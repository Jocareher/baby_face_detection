from typing import Tuple

import torch
import numpy as np
import math


def xyxyxyxy2xywhr(
    obb: torch.Tensor, angle: torch.Tensor, image_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Converts an oriented bounding box (OBB) from 8-point corner format (xyxyxyxy) and a given rotation angle
    to the (x_center, y_center, width, height, rotation) format (xywhr) in **absolute pixel coordinates**.

    This function supports both normalized and unnormalized input coordinates. If the coordinates appear
    to be normalized (i.e., values in [0, 1]), they are automatically scaled using the provided image size.

    Args:
        obb (torch.Tensor): Tensor of shape (N, 8) or (8,) containing 4 corner points of the OBB in the format:
                            [x1, y1, x2, y2, x3, y3, x4, y4].
        angle (torch.Tensor): Tensor of shape (N,) or scalar with angles in radians, one per OBB.
        image_size (Tuple[int, int]): Size of the image as (width, height).

    Returns:
        torch.Tensor: Tensor of shape (N, 5) in the format (x_center, y_center, width, height, angle),
                      all in pixel units.
    """
    # Check if obb is empty
    if obb.numel() == 0:
        return obb.new_empty((0, 5))

    # Ensure obb is a 2D tensor
    if obb.ndim == 1:
        obb = obb.unsqueeze(0)
    N = obb.shape[0]
    device = obb.device
    W, H = image_size

    # Automatically scale if coordinates are normalized
    if obb.max() <= 1.0:
        # Scale the coordinates to pixel space
        # The scale is applied to each coordinate (x, y) of the 4 corners
        # The scale is repeated for each corner (4 times)
        scale = torch.tensor([W, H] * 4, device=device, dtype=obb.dtype)
        obb_pix = obb * scale
    else:
        obb_pix = obb

    obb_pix = obb_pix.view(N, 4, 2)  # Convert flat to (N, 4, 2)
    center = obb_pix.mean(dim=1)  # Compute box center from corner average

    p0, p1, p2 = obb_pix[:, 0], obb_pix[:, 1], obb_pix[:, 2]
    width = (p1 - p0).norm(dim=1)  # Width: distance between point 0 and 1
    height = (p2 - p1).norm(dim=1)  # Height: distance between point 1 and 2

    angle_tensor = angle.to(device).float().view(-1, 1)  # Ensure shape (N, 1)

    return torch.cat(
        [center, width.unsqueeze(1), height.unsqueeze(1), angle_tensor], dim=1
    )


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
            torch.zeros(N, dtype=torch.long, device=anchors_xywhr.device),  # best_gt
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
    angles: torch.Tensor,  # (N,) — Anchor box rotation angles in radians
    image_size: Tuple[int, int],  # (W, H)
    scale: float = 0.5,  # Scale factor for offsets
) -> torch.Tensor:
    """
    Decodes predicted normalized vertex offsets into absolute OBB vertex coordinates.

    The function takes predicted deltas (offsets) for each of the 4 vertices of an oriented bounding box (OBB),
    which are normalized (typically via tanh to [-1, 1]), and reconstructs the absolute vertex positions
    in pixel coordinates by applying these offsets to the anchor box vertices. Optionally, the offsets
    are scaled by the anchor's diagonal length for scale invariance.

    Args:
        deltas (torch.Tensor): Tensor of shape (N, 8) containing predicted normalized offsets for each
            of the 4 vertices (x, y) of the OBB, typically in [-1, 1].
        anchors (torch.Tensor): Tensor of shape (N, 8) with anchor box vertex coordinates in pixel space,
            ordered as (x0, y0, x1, y1, x2, y2, x3, y3).
        angles (torch.Tensor): Tensor of shape (N,) with anchor box rotation angles in radians.
        image_size (Tuple[int, int]): Tuple (width, height) specifying the image dimensions for clamping.
        scale (float, optional): Scaling factor applied to the offset magnitude when use_diag is True.
            Default is 0.5.

    Returns:
        torch.Tensor: Tensor of shape (N, 8) containing the decoded absolute vertex positions for each OBB,
            clamped to the image bounds.
    """
    W, H = image_size  # Unpack image width and height

    N = anchors.size(0)  # Number of boxes

    verts = anchors.view(N, 4, 2)  # Reshape anchors to (N, 4, 2) for 4 vertices per box
    center = verts.mean(
        dim=1, keepdim=True
    )  # Compute box center for each anchor, shape (N, 1, 2)

    # Compute cosine and sine of the anchor angles for rotation matrix
    cos_t = angles.cos().view(N, 1, 1)  # (N, 1, 1)
    sin_t = angles.sin().view(N, 1, 1)  # (N, 1, 1)

    # Build rotation matrices for each box: shape (N, 2, 2)
    R = torch.cat(
        [
            torch.cat([cos_t, -sin_t], dim=2),  # First row: [cos, -sin]
            torch.cat([sin_t, cos_t], dim=2),  # Second row: [sin,  cos]
        ],
        dim=1,
    )

    # Compute the diagonal length of each anchor box (distance between vertex 0 and 2)
    diag = ((verts[:, 0] - verts[:, 2]).pow(2).sum(1).sqrt() * scale).view(
        N, 1, 1
    )  # (N, 1, 1)

    # Center the vertices and apply rotation (if needed)
    verts_rot = torch.bmm(verts - center, R) + center  # (N, 4, 2)

    # Apply the predicted deltas, scaled by the diagonal (if use_diag is True)
    verts_dec = verts_rot + deltas.view(N, 4, 2) * diag  # (N, 4, 2)

    # Clamp the decoded vertices to image bounds
    verts_dec[..., 0].clamp_(0, W)
    verts_dec[..., 1].clamp_(0, H)

    # Return the decoded vertices as (N, 8) tensor
    return verts_dec.view(N, 8)


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
    scale: float = 0.5,  # Scale factor for normalization
) -> torch.Tensor:
    """
    Encodes ground truth oriented bounding boxes (OBBs) as normalized deltas
    relative to the anchor boxes, for use in regression-based OBB heads.

    For each OBB, the function computes the difference between each ground truth
    vertex and the corresponding anchor vertex, normalizes this difference by the
    diagonal length of the anchor box (between vertex 0 and 2), and optionally
    applies a rotation to the anchor vertices to align with the ground truth angle.

    The output deltas are clamped to [-1, 1] and can be used as regression targets.

    Args:
        gt_boxes (torch.Tensor): Ground truth boxes in absolute vertex format (N, 8).
        anchors (torch.Tensor): Anchor boxes in absolute vertex format (N, 8).
        gt_angles (torch.Tensor): Rotation angles of ground truth OBBs in radians (N,).
        scale (float, optional): Scale factor for normalization. Default is 0.5.

    Returns:
        torch.Tensor: Normalized deltas of shape (N, 8), where each value is in Δx/Δy space,
                      clamped to [-1, 1].
    """
    N = anchors.size(0)  # Number of boxes

    verts_a = anchors.view(
        N, 4, 2
    )  # Reshape anchors to (N, 4, 2) for 4 vertices per box

    center = verts_a.mean(
        dim=1, keepdim=True
    )  # Compute anchor box center, shape (N, 1, 2)

    # Compute cosine and sine of the ground truth angles for rotation matrix
    cos_t = gt_angles.cos().view(N, 1, 1)  # (N, 1, 1)
    sin_t = gt_angles.sin().view(N, 1, 1)  # (N, 1, 1)

    # Build rotation matrices for each box: shape (N, 2, 2)
    R = torch.cat(
        [
            torch.cat([cos_t, -sin_t], dim=2),  # First row: [cos, -sin]
            torch.cat([sin_t, cos_t], dim=2),  # Second row: [sin,  cos]
        ],
        dim=1,
    )

    # Rotate anchor vertices to align with ground truth angle
    verts_rot = torch.bmm(verts_a - center, R) + center  # (N, 4, 2)

    # Compute the diagonal length of each anchor box (distance between vertex 0 and 2), scaled
    diag = ((verts_a[:, 0] - verts_a[:, 2]).pow(2).sum(1).sqrt() * scale).view(
        N, 1, 1
    )  # (N, 1, 1)

    # Compute normalized deltas between ground truth and rotated anchor vertices
    deltas = (gt_boxes.view(N, 4, 2) - verts_rot) / diag  # (N, 4, 2)

    # Flatten to (N, 8) and clamp to [-1, 1] for stability
    return deltas.view(N, 8).clamp_(-1.0, 1.0)
