from typing import List, Tuple

import torch
from shapely.geometry import Polygon


def compute_obb_iou_matrix(anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes an IoU matrix between all anchors (B, N, 8) and all GTs (B, M, 8).
    Each OBB is in the format [x1, y1, ..., x4, y4].

    Args:
        anchors (torch.Tensor): Tensor of anchor boxes (B, N, 8).
        gt_boxes (torch.Tensor): Tensor of ground truth boxes (B, M, 8).

    Returns:
        torch.Tensor: IoU matrix (B, N, M).
    """
    B, N, _ = anchors.shape  # Get batch size, number of anchors, and coordinates per OBB.
    M = gt_boxes.shape[1]  # Get number of ground truth boxes.
    ious = torch.zeros((B, N, M), dtype=torch.float32)  # Initialize IoU matrix with zeros.

    for b in range(B):  # Iterate through each image in the batch.
        for i in range(N):  # Iterate through each anchor box.
            anchor_poly = Polygon(anchors[b, i].view(4, 2).cpu().numpy())  # Create Shapely polygon for the anchor.
            if not anchor_poly.is_valid:  # Check if the anchor polygon is valid.
                continue  # Skip to the next anchor if not valid.
            for j in range(M):  # Iterate through each ground truth box.
                gt_poly = Polygon(gt_boxes[b, j].view(4, 2).cpu().numpy())  # Create Shapely polygon for the ground truth.
                if not gt_poly.is_valid:  # Check if the ground truth polygon is valid.
                    continue  # Skip to the next ground truth if not valid.
                inter = anchor_poly.intersection(gt_poly).area  # Calculate intersection area between anchor and ground truth.
                union = anchor_poly.union(gt_poly).area  # Calculate union area between anchor and ground truth.
                if union > 0:  # Check if union area is greater than zero.
                    ious[b, i, j] = inter / union  # Calculate IoU and store it in the matrix.
    return ious  # Return the IoU matrix.


def match_anchors_to_targets(anchors: torch.Tensor, gt_boxes: torch.Tensor, gt_classes: torch.Tensor,
                             gt_angles: torch.Tensor, iou_thresh: float = 0.5) -> Tuple[List[torch.Tensor],
                                                                                      List[torch.Tensor],
                                                                                      List[torch.Tensor],
                                                                                      List[torch.Tensor]]:
    """
    Matches anchors to targets using IoU. Only returns positive matches.

    Args:
        anchors (torch.Tensor): Anchor boxes (B, N, 8).
        gt_boxes (torch.Tensor): Ground truth boxes (B, M, 8).
        gt_classes (torch.Tensor): Ground truth classes (B, M).
        gt_angles (torch.Tensor): Ground truth angles (B, M).
        iou_thresh (float): IoU threshold for positive matches. Defaults to 0.5.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
            - matched_idx: Indices of positive anchors per batch (List[Tensor]).
            - matched_classes: Classes associated with each positive anchor.
            - matched_boxes: OBB targets per positive anchor.
            - matched_angles: Angles per positive anchor.
    """
    iou_matrix = compute_obb_iou_matrix(anchors, gt_boxes)  # (B, N, M)  # Calculate IoU matrix.

    matched_idx = []  # List to store indices of positive anchors.
    matched_classes = []  # List to store classes associated with positive anchors.
    matched_boxes = []  # List to store OBB targets associated with positive anchors.
    matched_angles = []  # List to store angles associated with positive anchors.

    B, N, M = iou_matrix.shape  # Get batch size, number of anchors, and number of ground truth boxes.

    for b in range(B):  # Iterate through each image in the batch.
        ious_b = iou_matrix[b]  # (N, M)  # Get IoU matrix for the current image.
        max_iou, gt_idx = ious_b.max(dim=1)  # best match GT for each anchor  # Find best ground truth for each anchor.
        pos_mask = max_iou > iou_thresh  # Create boolean mask for positive anchors.
        pos_idx = torch.where(pos_mask)[0]  # Get indices of positive anchors.

        matched_idx.append(pos_idx)  # Store indices of positive anchors.
        matched_classes.append(gt_classes[b][gt_idx[pos_idx]])  # Store classes associated with positive anchors.
        matched_boxes.append(gt_boxes[b][gt_idx[pos_idx]])  # Store OBB targets associated with positive anchors.
        matched_angles.append(gt_angles[b][gt_idx[pos_idx]])  # Store angles associated with positive anchors.

    return matched_idx, matched_classes, matched_boxes, matched_angles