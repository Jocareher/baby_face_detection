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


def compute_obb_iou_matrix_xywhr(
    anchors_xywhr: torch.Tensor, gt_xywhr: torch.Tensor
) -> torch.Tensor:
    """
    Computes the pairwise probabilistic IoU (pIoU) between two sets of oriented bounding boxes in xywhr format.

    This implementation is based on the probabilistic IoU from:
    "Learning Probabilistic Oriented Object Detection via Gaussian Distribution" (CVPR 2021).

    Args:
        anchors_xywhr (torch.Tensor): Tensor of shape (M, 5) with predicted OBBs in (x, y, w, h, θ) format.
        gt_xywhr (torch.Tensor): Tensor of shape (N, 5) with ground truth OBBs in (x, y, w, h, θ) format.

    Returns:
        torch.Tensor: IoU similarity matrix of shape (N, M) where entry (i, j) is the similarity between
                      gt i and anchor j.
    """

    return batch_probiou(anchors_xywhr, gt_xywhr)


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
    iou_thr: float = 0.5,  # IoU threshold for positive match
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
        iou_thr (float, optional): IoU threshold to determine positive matches. Default is 0.5.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - A boolean tensor of shape (N,) indicating which anchors are matched (positive).
            - A tensor of shape (N,) with the index of the best GT box for each anchor.
    """
    W, H = image_size
    gt_xywhr = xyxyxyxy2xywhr(gt_boxes_xy, gt_angles, (W, H))

    # Compute pairwise pIoU between anchors and GT boxes
    iou_matrix = compute_obb_iou_matrix_xywhr(anchors_xywhr, gt_xywhr)
    iou, best_gt = iou_matrix.max(dim=1)  # Best match for each anchor

    return iou > iou_thr, best_gt


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
    image_size: Tuple[int, int],  # (W, H)
    use_diag: bool = True,  # Whether to scale offsets by the anchor's diagonal
) -> torch.Tensor:
    """
    Decodes predicted OBB vertex offsets back to absolute vertex coordinates.

    The deltas are learned displacements for each of the 4 vertices of an OBB, normalized
    to [-1, 1] via a tanh activation. This function rescales and shifts these deltas
    with respect to the given anchor boxes.

    Args:
        deltas (torch.Tensor): Offset predictions of shape (N, 8), in the range [-1, 1].
        anchors (torch.Tensor): Anchor box vertices of shape (N, 8), in pixel coordinates.
        image_size (Tuple[int, int]): (width, height) of the image to clamp the results.
        use_diag (bool, optional):
            If True, offsets are scaled by the average diagonal of the anchor box
            (i.e., roughly its "natural" size). If False, the offsets are interpreted
            as pixel values directly. Default is True.

    Returns:
        torch.Tensor: Decoded absolute vertex positions of shape (N, 8), clamped to image bounds.
    """
    W, H = image_size

    if use_diag:
        # Use average diagonal length as a scale factor (per box)
        diag = anchors.view(-1, 4, 2).std(dim=1).mean(dim=1, keepdim=True)  # (N, 1)
        verts = anchors + deltas * diag  # Displace each vertex up to ±diag pixels
    else:
        # Direct displacement in pixel space
        verts = anchors + deltas  # Displace only ±1 px per dimension

    # Clamp decoded vertices to valid image bounds
    clamp_vec = verts.new_tensor([W, H] * 4)  # (8,)
    verts = torch.clamp_min(verts, 0.0)  # Lower bound: 0
    verts = torch.minimum(verts, clamp_vec)  # Upper bound: image size

    return verts


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
    #
    cx, cy, w, h, angle = xywhr.T
    dx, dy = w / 2, h / 2

    # Initial corners before rotation
    corners = torch.stack(
        [
            torch.stack([-dx, -dy], dim=1),
            torch.stack([dx, -dy], dim=1),
            torch.stack([dx, dy], dim=1),
            torch.stack([-dx, dy], dim=1),
        ],
        dim=1,
    )

    # Rotation matrix
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    # Construct the rotation matrix
    rot_matrix = torch.stack(
        [torch.stack([cos_a, -sin_a], dim=1), torch.stack([sin_a, cos_a], dim=1)], dim=2
    )

    # Rotate the corners
    # corners: (N, 4, 2)
    # rot_matrix: (N, 2, 2)
    # Perform batch matrix multiplication
    # Note: corners are in shape (N, 4, 2) and rot_matrix is in shape (N, 2, 2)
    # The result is (N, 4, 2) after rotation
    # and translation
    rotated = torch.bmm(corners, rot_matrix)
    centers = torch.stack([cx, cy], dim=1)[:, None, :]
    return (rotated + centers).detach().cpu().numpy()
