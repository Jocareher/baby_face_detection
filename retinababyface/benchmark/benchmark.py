import math
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
from matplotlib import pyplot as plt

from loss.utils import verts_to_xywhr_with_theta, batch_probiou


def read_infantface_gt_xywhr(
    gt_txt_path: Path,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lee GT del InfantFace (formato por línea: x1 y1 x2 y2 en PIXELES).
    Devuelve:
      - xywhr_gt: (N,5) en pixeles (cx,cy,w,h,theta) con theta=0.0
      - cls_gt:  (N,) dummy con todo a 0 (no se usa en loc-only)
    Si el archivo no existe o está vacío, retorna tensores vacíos.
    """
    if not gt_txt_path.exists():
        return torch.empty((0, 5), dtype=torch.float32), torch.empty(
            (0,), dtype=torch.long
        )

    xywhr_list = []
    with open(gt_txt_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) < 4:
                continue
            x1 = float(toks[0])
            y1 = float(toks[1])
            x2 = float(toks[2])
            y2 = float(toks[3])
            x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
            y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
            w = max(0.0, x_max - x_min)
            h = max(0.0, y_max - y_min)
            if w <= 0.0 or h <= 0.0:
                continue
            cx = x_min + w * 0.5
            cy = y_min + h * 0.5
            theta = 0.0  # AABB
            xywhr_list.append([cx, cy, w, h, theta])

    if len(xywhr_list) == 0:
        return torch.empty((0, 5), dtype=torch.float32), torch.empty(
            (0,), dtype=torch.long
        )

    xywhr = torch.tensor(xywhr_list, dtype=torch.float32)
    cls_dummy = torch.zeros((xywhr.size(0),), dtype=torch.long)  # no se usa en loc-only
    return xywhr, cls_dummy


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


def read_retinababyface_preds_xywhr(
    pred_txt_path: Path,
    min_score: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reads predictions exported by RetinaBabyFace and converts them to a consistent format.

    The input file is expected to have lines formatted as:
        class_idx x1 y1 x2 y2 x3 y3 x4 y4 angle_rads score
    where:
        - class_idx: Class index (integer, 0-based).
        - x1, y1, ..., x4, y4: Absolute pixel coordinates of the bounding box vertices.
        - angle_rads: Rotation angle of the bounding box in radians.
        - score: Confidence score of the prediction (float).

    Args:
        pred_txt_path (Path): Path to the predictions file.
        min_score (float): Minimum score threshold for filtering predictions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - pred_xywhr: Tensor of shape (P, 5) containing bounding boxes in pixel coordinates
              as (cx, cy, w, h, θ) where θ is in radians.
            - pred_cls: Tensor of shape (P,) containing class indices for each prediction.
            - pred_scores: Tensor of shape (P,) containing confidence scores for each prediction.

    If the file does not exist or contains no valid predictions, returns empty tensors.
    """
    if not pred_txt_path.exists():
        # Return empty tensors if the file does not exist
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
            # Expecting 11 tokens: class_idx + 8 coordinates + angle + score
            if len(toks) < 11:
                continue  # Skip malformed lines

            try:
                # Parse class index, bounding box vertices, angle, and score
                cls_idx = int(float(toks[0]))
                x1, y1 = float(toks[1]), float(toks[2])
                x2, y2 = float(toks[3]), float(toks[4])
                x3, y3 = float(toks[5]), float(toks[6])
                x4, y4 = float(toks[7]), float(toks[8])
                theta = float(toks[9])  # Rotation angle in radians
                score = float(toks[10])  # Confidence score
            except ValueError:
                # Skip lines with invalid numeric values
                continue

            if score < min_score:
                continue  # Skip predictions below the score threshold

            # Compute the center of the bounding box as the average of the vertices
            cx = (x1 + x2 + x3 + x4) * 0.25
            cy = (y1 + y2 + y3 + y4) * 0.25

            # Compute the width as the distance between vertices v0 and v1
            # Compute the height as the distance between vertices v1 and v2
            def dist(ax, ay, bx, by):
                dx, dy = (bx - ax), (by - ay)
                return math.hypot(dx, dy)

            w = dist(x1, y1, x2, y2)
            h = dist(x2, y2, x3, y3)

            # Skip degenerate bounding boxes with non-positive width or height
            if w <= 0.0 or h <= 0.0:
                continue

            # Append the bounding box, class index, and score to their respective lists
            xywhr_list.append(torch.tensor([cx, cy, w, h, theta], dtype=torch.float32))
            cls_list.append(cls_idx)
            score_list.append(score)

    if not xywhr_list:
        # Return empty tensors if no valid predictions were found
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


def read_pcn_preds_xywhr(
    pred_txt_path: Path,
    img_wh: Tuple[int, int],
    min_score: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reads PCN (Progressive Calibration Networks) predictions from a text file and converts them to a consistent format.

    The input file is expected to have lines formatted as:
        x1 y1 x2 y2 angle_degrees score
    where:
        - x1, y1: Top-left corner of the bounding box (in pixels).
        - x2, y2: Bottom-right corner of the bounding box (in pixels).
        - angle_degrees: Rotation angle of the bounding box in degrees.
        - score: Confidence score of the prediction (float).

    Args:
        pred_txt_path (Path): Path to the predictions file.
        img_wh (Tuple[int, int]): Width and height of the image in pixels.
        min_score (float): Minimum score threshold for filtering predictions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - pred_xywhr: Tensor of shape (P, 5) containing bounding boxes in pixel coordinates
              as (cx, cy, w, h, θ) where θ is in radians.
            - pred_scores: Tensor of shape (P,) containing confidence scores for each prediction.

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
            # Ensure the line has at least 6 values (x1, y1, x2, y2, angle, score)
            if len(toks) < 6:
                continue
            try:
                # Parse bounding box coordinates, angle, and score
                x1 = float(toks[0])
                y1 = float(toks[1])
                x2 = float(toks[2])
                y2 = float(toks[3])
                ang_deg = float(toks[4])  # Angle in degrees
                score = float(toks[5])  # Confidence score
            except ValueError:
                # Skip lines with invalid numeric values
                continue

            if score < min_score:
                continue  # Skip predictions below the score threshold

            # Clamp coordinates to image boundaries and ensure proper ordering
            x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
            y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
            x_min = max(0.0, min(x_min, W0 - 1))
            y_min = max(0.0, min(y_min, H0 - 1))
            x_max = max(0.0, min(x_max, W0 - 1))
            y_max = max(0.0, min(y_max, H0 - 1))

            # Compute bounding box dimensions
            w = max(0.0, x_max - x_min)  # Width
            h = max(0.0, y_max - y_min)  # Height
            if w <= 0.0 or h <= 0.0:
                continue  # Skip invalid boxes

            # Compute the center of the bounding box
            cx = x_min + w / 2.0
            cy = y_min + h / 2.0

            # Convert angle from degrees to radians
            theta = math.radians(ang_deg)
            # Normalize angle to the range [-π, π]
            if theta > math.pi or theta < -math.pi:
                theta = ((theta + math.pi) % (2 * math.pi)) - math.pi

            # Append the bounding box and score to their respective lists
            xywhr_list.append(
                torch.tensor([cx, cy, w, h, float(theta)], dtype=torch.float32)
            )
            score_list.append(float(score))

    if not xywhr_list:
        # Return empty tensors if no valid predictions were found
        return (
            torch.empty((0, 5), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
        )

    # Stack results into tensors
    return torch.stack(xywhr_list, dim=0), torch.tensor(score_list, dtype=torch.float32)


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


def count_adults_in_gt(gt_txt_path: Path) -> int:
    """
    Counts the number of annotations in the ground truth file where `child_prob` is 0 (adults).

    The ground truth file is expected to have lines formatted as:
        cls_idx child_prob x1 y1 x2 y2 x3 y3 x4 y4 angle
    where:
        - cls_idx: Class index (integer, 0-based).
        - child_prob: Indicator (1 for baby, 0 for adult).
        - x1, y1, ..., x4, y4: Normalized coordinates of the bounding box vertices.
        - angle: Rotation angle in radians.

    Args:
        gt_txt_path (Path): Path to the ground truth annotation file.

    Returns:
        int: The number of annotations with `child_prob` equal to 0 (adults).
             Returns 0 if the file does not exist or contains no valid annotations.
    """
    if not gt_txt_path.exists():
        return 0  # Return 0 if the file does not exist

    n_adults = 0  # Counter for adult annotations
    with open(gt_txt_path, "r") as f:
        for line in f:
            toks = line.strip().split()  # Split the line into tokens
            if not toks:
                continue  # Skip empty lines

            # Check if the line has at least two tokens and the second token is a digit
            if len(toks) >= 2 and toks[1].isdigit():
                if int(toks[1]) == 0:  # Check if `child_prob` is 0 (adult)
                    n_adults += 1  # Increment the counter for adults

    return n_adults


def compute_loc_curves_from_predictions(
    y_is_tp: List[int], y_scores: List[float], n_gt: int, n_steps: int = 200
) -> Dict[str, Any]:
    """
    Computes localization performance curves (precision, recall, F1-score)
    as a function of score thresholds.

    This function evaluates the localization performance of a model by calculating
    precision, recall, and F1-score at various score thresholds. It also identifies
    the threshold that maximizes the F1-score.

    Args:
        y_is_tp (List[int]): A list of binary values (1 for true positive, 0 for false positive)
                             indicating whether each prediction is a true positive.
        y_scores (List[float]): A list of confidence scores corresponding to the predictions.
        n_gt (int): The total number of ground truth objects.
        n_steps (int, optional): The number of thresholds to evaluate between 0 and 1.
                                 Defaults to 200.

    Returns:
        Dict[str, Any]: A dictionary containing the following keys:
            - "thresholds": Array of evaluated thresholds.
            - "precision": Array of precision values at each threshold.
            - "recall": Array of recall values at each threshold.
            - "f1": Array of F1-scores at each threshold.
            - "best_idx": Index of the threshold with the highest F1-score.
            - "best_th": Threshold value that maximizes the F1-score.
            - "best_P": Precision at the best threshold.
            - "best_R": Recall at the best threshold.
            - "best_F1": Maximum F1-score achieved.

    If there are no predictions or the number of ground truth objects is zero,
    the function returns empty arrays and default values.
    """
    # Convert inputs to numpy arrays for efficient computation
    scores = np.asarray(y_scores, dtype=np.float32)
    is_tp = np.asarray(y_is_tp, dtype=np.int32)

    # Handle edge cases where there are no predictions or no ground truth objects
    if scores.size == 0 or n_gt <= 0:
        z = np.array([])
        return {
            "thresholds": z,
            "precision": z,
            "recall": z,
            "f1": z,
            "best_idx": -1,
            "best_th": 0.0,
            "best_P": 0.0,
            "best_R": 0.0,
            "best_F1": 0.0,
        }

    # Generate evenly spaced thresholds between 0 and 1
    thresholds = np.linspace(0.0, 1.0, n_steps)

    # Initialize lists to store precision, recall, and F1-score at each threshold
    precs, recs, f1s = [], [], []

    # Iterate over each threshold and compute precision, recall, and F1-score
    for t in thresholds:
        # Keep predictions with scores greater than or equal to the current threshold
        keep = scores >= t

        # Count true positives (TP) and false positives (FP) for the current threshold
        tp = int((is_tp[keep] == 1).sum())  # True positives
        fp = int((is_tp[keep] == 0).sum())  # False positives

        # Compute precision (P), recall (R), and F1-score (F1)
        P = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0  # Precision
        R = (tp / n_gt) if n_gt > 0 else 0.0  # Recall
        F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0  # F1-score

        # Append the computed values to their respective lists
        precs.append(P)
        recs.append(R)
        f1s.append(F1)

    # Convert lists to numpy arrays for consistency
    precs = np.asarray(precs)
    recs = np.asarray(recs)
    f1s = np.asarray(f1s)

    # Identify the index of the threshold that maximizes the F1-score
    best_idx = int(f1s.argmax()) if f1s.size > 0 else -1

    # Return the computed metrics and the best threshold information
    return {
        "thresholds": thresholds,
        "precision": precs,
        "recall": recs,
        "f1": f1s,
        "best_idx": best_idx,
        "best_th": float(thresholds[best_idx]) if best_idx >= 0 else 0.0,
        "best_P": float(precs[best_idx]) if best_idx >= 0 else 0.0,
        "best_R": float(recs[best_idx]) if best_idx >= 0 else 0.0,
        "best_F1": float(f1s[best_idx]) if best_idx >= 0 else 0.0,
    }


def plot_precision_recall_vs_threshold(th, prec, rec, best_th=None, out_path=None):
    """
    Plots precision and recall as functions of the score threshold.

    This function generates a plot showing how precision and recall vary with the
    score threshold. It optionally highlights the best threshold (e.g., the one
    that maximizes F1-score) and saves the plot to a file or returns the figure object.

    Args:
        th (array-like): Array of threshold values.
        prec (array-like): Array of precision values corresponding to the thresholds.
        rec (array-like): Array of recall values corresponding to the thresholds.
        best_th (float, optional): The threshold value to highlight on the plot.
                                    Defaults to None.
        out_path (str or Path, optional): Path to save the plot as an image file.
                                          If None, the function returns the figure object.
                                          Defaults to None.

    Returns:
        matplotlib.figure.Figure: The generated plot as a figure object if `out_path` is None.
                                  Otherwise, the plot is saved to the specified path, and
                                  the function returns None.
    """
    # Create a new figure and axis for the plot
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    # Plot precision and recall curves
    ax.plot(th, prec, label="Precision", color="blue")
    ax.plot(th, rec, label="Recall", color="orange")

    # Set axis limits and labels
    ax.set_xlim(0, 1)  # Thresholds range from 0 to 1
    ax.set_ylim(0, 1.05)  # Precision and recall values range from 0 to 1
    ax.set_xlabel("Threshold")  # X-axis label
    ax.set_ylabel("Score")  # Y-axis label
    ax.set_title("Precision and Recall vs. Threshold (Localization-only)")  # Plot title

    # Highlight the best threshold if provided
    if best_th is not None:
        ax.axvline(best_th, linestyle="--", linewidth=1, color="gray")  # Vertical line
        ax.text(
            best_th, 1.02, f"best={best_th:.3f}", ha="center", va="bottom", fontsize=9
        )  # Annotate the best threshold

    # Add a legend to the plot
    ax.legend(loc="best")

    # Adjust layout for better appearance
    fig.tight_layout()

    # Save the plot to a file or return the figure object
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")  # Save to file
        plt.close(fig)  # Close the figure to free memory
    else:
        return fig  # Return the figure object


def read_raw_gt_lines(path: Path) -> List[str]:
    """
    Reads all non-empty lines from a ground truth (GT) file.

    Args:
        path (Path): Path to the ground truth file.

    Returns:
        List[str]: A list of non-empty, stripped lines from the file.
                   Returns an empty list if the file does not exist.
    """
    if not path.exists():
        return []  # Return an empty list if the file does not exist
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]  # Read and strip non-empty lines


def classify_image_gt(gt_path: Path) -> str:
    """
    Classifies the type of annotations in a ground truth (GT) file.

    The function determines whether the GT file corresponds to:
        - "BABY": If there is at least one annotation for a baby (class indices 0-4).
        - "ADULT_ONLY": If there are annotations, but all are for adults (class index -1).
        - "BG": If the file does not exist, is empty, or contains no valid annotations.

    Args:
        gt_path (Path): Path to the ground truth annotation file.

    Returns:
        str: One of the following classification labels:
             - "BABY": At least one baby annotation is present.
             - "ADULT_ONLY": Only adult annotations are present.
             - "BG": No annotations or invalid file.
    """
    # Read all non-empty lines from the GT file
    lines = read_raw_gt_lines(gt_path)
    if not lines:
        return "BG"  # Return "BG" if the file is empty or does not exist

    has_baby = False  # Flag to check if there is at least one baby annotation
    only_adult = True  # Flag to check if all annotations are for adults

    # First pass: Check for baby annotations (class indices 0-4)
    for ln in lines:
        toks = ln.split()
        if not toks:
            continue  # Skip empty or malformed lines
        try:
            cls_idx = int(float(toks[0]))  # Parse the class index
        except Exception:
            continue  # Skip lines with invalid class indices
        if cls_idx != -1:  # Class indices 0-4 correspond to babies
            has_baby = True
            only_adult = False
            break  # Exit early if a baby annotation is found

    if has_baby:
        return "BABY"  # Return "BABY" if at least one baby annotation is found

    # Second pass: Check for adult annotations (class index -1)
    for ln in lines:
        toks = ln.split()
        try:
            cls_idx = int(float(toks[0]))  # Parse the class index
        except Exception:
            continue  # Skip lines with invalid class indices
        if cls_idx == -1:  # Class index -1 corresponds to adults
            return "ADULT_ONLY"  # Return "ADULT_ONLY" if at least one adult annotation is found

    # If no valid annotations are found, treat the file as background (BG)
    return "BG"
