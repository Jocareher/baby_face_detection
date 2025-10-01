import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List
from loss.utils import verts_to_xywhr_with_theta, batch_probiou


def read_gt_baby_xywhr(
    gt_txt_path: Path, img_wh: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reads ground truth (GT) annotations for baby detections from a text file.

    The input file is expected to have lines formatted as:
        cls_idx child_prob x1 y1 x2 y2 x3 y3 x4 y4 theta
    where:
        - cls_idx: Class index (integer, 0-based).
        - child_prob: Indicator (1 for baby, 0 for others).
        - x1, y1, ..., x4, y4: Normalized coordinates of the bounding box vertices.
        - theta: Rotation angle in radians.

    Args:
        gt_txt_path (Path): Path to the ground truth annotation file.
        img_wh (Tuple[int, int]): Width and height of the image in pixels.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - xywhr_gt: Tensor of shape (N, 5) containing bounding boxes in pixel coordinates
              as (cx, cy, w, h, θ) for baby detections.
            - cls_gt: Tensor of shape (N,) containing class indices for each detection.

    If the file does not exist or contains no valid baby detections, returns empty tensors.
    """
    W0, H0 = img_wh
    if not gt_txt_path.exists():
        return torch.empty((0, 5), dtype=torch.float32), torch.empty(
            (0,), dtype=torch.long
        )

    xywhr_list, cls_list = [], []
    with open(gt_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            toks = line.split()
            cls_idx = int(toks[0])  # Class index
            child_prob = int(toks[1])  # Baby indicator (1 for baby, 0 for others)

            if child_prob != 1:
                continue  # Ignore non-baby annotations

            # Extract normalized bounding box vertices (x1, y1, ..., x4, y4)
            pts = [float(t) for t in toks[2 : 2 + 8]]
            pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
            # Convert normalized coordinates to pixel coordinates
            pts[:, 0] *= W0
            pts[:, 1] *= H0

            # Extract rotation angle (theta) in radians
            theta = float(toks[10])

            # Convert vertices and angle to (cx, cy, w, h, θ) format
            verts_t = torch.from_numpy(pts.reshape(1, 4, 2))
            theta_t = torch.tensor([theta], dtype=torch.float32)
            xywhr_t = verts_to_xywhr_with_theta(
                verts_t, theta_t
            )  # Use utility function
            xywhr_list.append(xywhr_t[0])
            cls_list.append(cls_idx)

    if len(xywhr_list) == 0:
        return torch.empty((0, 5), dtype=torch.float32), torch.empty(
            (0,), dtype=torch.long
        )

    # Stack results into tensors
    return torch.stack(xywhr_list, dim=0), torch.tensor(cls_list, dtype=torch.long)


def read_sota_preds_xywhr_xyxy(
    pred_txt_path: Path,
    img_wh: Tuple[int, int],
    min_score: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reads predictions from a text file and converts them to a consistent format.

    The input file can contain predictions in one of the following formats per line:
        1) x1 y1 x2 y2
        2) score x1 y1 x2 y2
        3) x1 y1 x2 y2 score
        4) score x1 y1 x2 y2 angle (angle is ignored; treated as AABB => θ=0)

    Args:
        pred_txt_path (Path): Path to the predictions file.
        img_wh (Tuple[int, int]): Width and height of the image in pixels.
        min_score (float): Minimum score threshold for filtering predictions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - pred_xywhr: Tensor of shape (M, 5) containing bounding boxes in pixel coordinates
              as (cx, cy, w, h, θ=0).
            - pred_scores: Tensor of shape (M,) containing scores for each prediction.

    If the file does not exist or contains no valid predictions, returns empty tensors.
    """
    W0, H0 = img_wh
    if not pred_txt_path.exists():
        return (
            torch.empty((0, 5), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
        )

    xywhr_list, score_list = [], []

    with open(pred_txt_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue  # Skip empty lines
            toks = line.split()
            vals = [float(t) for t in toks]

            # --- Detect layout and parse values ---
            score = 1.0  # Default score if not provided
            if len(vals) == 4:
                # Format: x1 y1 x2 y2
                x1, y1, x2, y2 = vals
            elif len(vals) >= 5:
                # Format: score x1 y1 x2 y2 or x1 y1 x2 y2 score
                # If the first value is a valid score, use it
                if 0.0 <= vals[0] <= 1.0 and vals[1] >= 0 and vals[3] >= 0:
                    score, x1, y1, x2, y2 = vals[:5]
                # If the last value is a valid score, use it
                elif 0.0 <= vals[-1] <= 1.0 and vals[0] >= 0 and vals[2] >= 0:
                    x1, y1, x2, y2, score = vals[:5]
                else:
                    # Fallback: assume the first four values are x1 y1 x2 y2
                    x1, y1, x2, y2 = vals[:4]
                    score = 1.0
            else:
                # Invalid line format
                continue

            if score < min_score:
                continue  # Skip predictions below the score threshold

            # --- Clip coordinates to image boundaries ---
            x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
            y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
            x_min = max(0.0, min(x_min, W0 - 1))
            y_min = max(0.0, min(y_min, H0 - 1))
            x_max = max(0.0, min(x_max, W0 - 1))
            y_max = max(0.0, min(y_max, H0 - 1))

            # --- Compute bounding box dimensions ---
            w = max(0.0, x_max - x_min)
            h = max(0.0, y_max - y_min)
            if w <= 0.0 or h <= 0.0:
                continue  # Skip invalid boxes

            # --- Convert to (cx, cy, w, h, θ=0) format ---
            cx = x_min + w / 2.0
            cy = y_min + h / 2.0
            theta = 0.0  # AABB => OBB with θ=0

            xywhr_list.append(torch.tensor([cx, cy, w, h, theta], dtype=torch.float32))
            score_list.append(score)

    if len(xywhr_list) == 0:
        return (
            torch.empty((0, 5), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
        )

    # Stack results into tensors
    return torch.stack(xywhr_list, dim=0), torch.tensor(score_list, dtype=torch.float32)


def read_yolo_oriented_preds_xywhr(
    pred_txt_path: Path,
    min_score: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reads YOLO-oriented predictions from a text file and converts them to a consistent format.

    The input file is expected to have lines formatted as:
        class_idx x1 y1 x2 y2 angle_radians score
    where:
        - class_idx: Class index (integer, 0-based).
        - x1, y1: Top-left corner of the bounding box (in pixels).
        - x2, y2: Bottom-right corner of the bounding box (in pixels).
        - angle_radians: Rotation angle of the bounding box in radians.
        - score: Confidence score of the prediction (float).

    Args:
        pred_txt_path (Path): Path to the predictions file.
        min_score (float): Minimum score threshold for filtering predictions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - pred_xywhr: Tensor of shape (P, 5) containing bounding boxes in pixel coordinates
              as (cx, cy, w, h, θ).
            - pred_cls: Tensor of shape (P,) containing class indices for each prediction.
            - pred_scores: Tensor of shape (P,) containing confidence scores for each prediction.

    If the file does not exist or contains no valid predictions, returns empty tensors.
    """
    if not pred_txt_path.exists():
        return (
            torch.empty((0, 5), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    xywhr_list, cls_list, score_list = [], [], []
    with open(pred_txt_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue  # Skip empty lines
            toks = line.split()
            if len(toks) < 7:
                # Skip malformed lines
                continue

            # Parse values from the line
            cls_idx = int(float(toks[0]))  # Class index
            x1 = float(toks[1])  # Top-left x-coordinate
            y1 = float(toks[2])  # Top-left y-coordinate
            x2 = float(toks[3])  # Bottom-right x-coordinate
            y2 = float(toks[4])  # Bottom-right y-coordinate
            theta = float(toks[5])  # Rotation angle in radians
            score = float(toks[6])  # Confidence score

            if score < min_score:
                continue  # Skip predictions below the score threshold

            # Normalize the order of corners and compute (cx, cy, w, h)
            x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
            y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
            w = max(0.0, x_max - x_min)  # Width of the bounding box
            h = max(0.0, y_max - y_min)  # Height of the bounding box
            if w <= 0.0 or h <= 0.0:
                continue  # Skip invalid boxes
            cx = x_min + w / 2.0  # Center x-coordinate
            cy = y_min + h / 2.0  # Center y-coordinate

            # Append the parsed values to their respective lists
            xywhr_list.append(torch.tensor([cx, cy, w, h, theta], dtype=torch.float32))
            cls_list.append(cls_idx)
            score_list.append(score)

    if len(xywhr_list) == 0:
        return (
            torch.empty((0, 5), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    # Stack results into tensors
    return (
        torch.stack(xywhr_list, dim=0),
        torch.tensor(cls_list, dtype=torch.long),
        torch.tensor(score_list, dtype=torch.float32),
    )


def greedy_match(
    gt_xywhr: torch.Tensor,
    pr_xywhr: torch.Tensor,
    pr_scores: torch.Tensor,
    iou_th: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Performs greedy matching between ground truth (GT) and predicted bounding boxes based on IoU.

    This function matches predictions to ground truth boxes using a greedy strategy,
    prioritizing predictions with higher confidence scores. A match is made if the IoU
    between a prediction and a ground truth box exceeds the specified threshold.

    Args:
        gt_xywhr (torch.Tensor): Tensor of shape (G, 5) containing ground truth bounding boxes
                                 in (cx, cy, w, h, θ) format.
        pr_xywhr (torch.Tensor): Tensor of shape (P, 5) containing predicted bounding boxes
                                 in (cx, cy, w, h, θ) format.
        pr_scores (torch.Tensor): Tensor of shape (P,) containing confidence scores for each prediction.
        iou_th (float): IoU threshold for matching predictions to ground truth boxes.

    Returns:
        Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
            - matches: List of tuples (gt_idx, pr_idx, iou) representing matched indices and their IoU.
            - unmatched_gt: List of indices of ground truth boxes that were not matched.
            - unmatched_pr: List of indices of predictions that were not matched.
    """
    # If either GT or predictions are empty, return empty results
    if gt_xywhr.numel() == 0 or pr_xywhr.numel() == 0:
        return [], list(range(len(gt_xywhr))), list(range(len(pr_xywhr)))

    # Compute IoU matrix (G x P) between all GT and predicted boxes
    ious = batch_probiou(gt_xywhr.to(pr_xywhr.device), pr_xywhr)

    # Sort predictions by confidence scores in descending order
    order = torch.argsort(pr_scores, descending=True)
    gt_taken = torch.zeros(len(gt_xywhr), dtype=torch.bool)  # Tracks matched GT boxes
    pr_taken = torch.zeros(
        len(pr_xywhr), dtype=torch.bool
    )  # Tracks matched predictions

    matches = []
    for j in order.tolist():
        # Find the best unmatched GT box for the current prediction
        col = ious[:, j].clone()
        col[gt_taken] = -1.0  # Ignore already matched GT boxes
        iou_val, i = col.max(0)  # Get the best IoU and corresponding GT index
        i = int(i.item())
        iou_val = float(iou_val.item())

        # Match if IoU exceeds the threshold and the GT box is not already matched
        if iou_val >= iou_th and not gt_taken[i]:
            gt_taken[i] = True
            pr_taken[j] = True
            matches.append((i, j, iou_val))

    # Collect indices of unmatched GT boxes and predictions
    unmatched_gt = [i for i, t in enumerate(gt_taken.tolist()) if not t]
    unmatched_pr = [j for j, t in enumerate(pr_taken.tolist()) if not t]

    return matches, unmatched_gt, unmatched_pr
