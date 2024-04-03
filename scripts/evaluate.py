import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def read_bboxes(file_path: str, format: str = "detection") -> list:
    """
    Reads bounding boxes from a text file and returns them as a list.

    Args:
        - file_path (str): Path to the text file containing bounding box coordinates.
        - format (str): Format of the bounding boxes. It can be 'detection' or 'ground_truth'.

    Returns:
        - list: List of bounding boxes, where each bounding box is represented as either a list [x1, y1, x2, y2] or a tuple ((x1, y1, x2, y2), score).

    Description:
        This function reads the bounding box coordinates from a text file and returns them as a list.
        If the format is 'ground_truth', it converts the coordinates from (x, y, w, h) format to (x1, y1, x2, y2) format to ensure consistency.
        If the format is 'detection', it expects the bounding boxes to be in either (x1, y1, x2, y2) format or (x1, y1, x2, y2, score) format, where score is the probability score associated with the detection.
    """
    # Open the text file containing bounding box coordinates
    with open(file_path, "r") as f:
        # Read all lines from the file
        lines = f.readlines()
    # Initialize an empty list to store bounding boxes
    bboxes = []
    # Check the format of bounding boxes
    if format == "ground_truth":
        # Iterate over lines and convert from c, (x, y, w, h) to (x1, y1, x2, y2)
        for line in lines:
            c, x, y, w, h = map(float, line.strip().split())
            bboxes.append([x, y, x + w, y + h])  # Convert to x1, y1, x2, y2 format
    else:
        # Iterate over lines and parse bounding box information
        for line in lines:
            # Split the line into parts
            parts = line.strip().split()
            # Check if each detection includes a probability score
            if len(parts) == 5:  # Has probability score
                # Extract the score and bounding box coordinates
                score = float(parts[0])
                bbox = list(map(float, parts[1:]))
                # Append bounding box and score as tuple to the list
                bboxes.append((bbox, score))
            else:
                # Extract bounding box coordinates
                bbox = list(map(float, parts))
                # Append bounding box to the list
                bboxes.append(bbox)  # For ground truth compatibility
    # Return the list of bounding boxes
    return bboxes


def iou_aabb(bbox1: list, bbox2: list) -> float:
    """
    Calculates the Intersection over Union (IoU) between two Axis-Aligned Bounding Boxes (AABB).

    Args:
        - bbox1 (list): Coordinates of the first bounding box [x1, y1, x2, y2].
        - bbox2 (list): Coordinates of the second bounding box [x1, y1, x2, y2].

    Returns:
        - float: IoU value between the two bounding boxes.

    Description:
        This function calculates the Intersection over Union (IoU) between two Axis-Aligned Bounding Boxes (AABB).
        It first determines the coordinates of the intersection rectangle and then calculates the area of intersection.
        Then, it calculates the IoU using the intersection area and the area of both bounding boxes.
    """
    # Calculate the (x, y) coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    # Check if there is no intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # Calculate the areas of both bounding boxes
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    # Calculate IoU
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou


def integrated_detection_metrics_and_roc(
    det_path: str, gt_path: str, title: str = "ROC curve"
) -> tuple:
    """
    Computes integrated detection metrics (precision, recall, F1-score) and plots the Receiver Operating Characteristic (ROC) curve for a set of detection results against ground truth annotations.

    This function calculates the following metrics:
        - Precision: The ratio of true positive detections to the total number of positive predictions made.
          It is calculated as TP / (TP + FP), where TP is the number of true positives (detections that correctly match a ground truth bounding box with IoU >= 0.5)
          and FP is the number of false positives (detections that do not correctly match any ground truth bounding box or match already detected ground truth bounding box).

        - Recall: The ratio of true positive detections to the total number of actual positives (ground truth bounding boxes).
          It is calculated as TP / (TP + FN), where FN is the number of false negatives (ground truth bounding boxes that were not detected by any detection bounding box).

        - F1-Score: The harmonic mean of precision and recall, providing a single metric to assess the balance between them.
          It is calculated as 2 * (precision * recall) / (precision + recall), giving equal weight to both precision and recall.

    Additionally, it plots the Receiver Operating Characteristic (ROC) curve, showing the trade-off between true positive rate (sensitivity) and false positive rate (1-specificity).

    Args:
        - det_path (str): Path to the directory containing detection bounding box files (.txt format), where each file contains bounding boxes for detections in an image.
        - gt_path (str): Path to the directory containing ground truth bounding box files (.txt format), where each file contains bounding boxes for actual objects in the corresponding image.
        - title (str): Title of the ROC curve plot. Default is "ROC curve".

    Returns:
        - tuple: Contains calculated precision, recall, and F1-score. Each metric is a float value between 0 and 1, where 1 indicates perfect precision, recall, or F1-score.

    Note:
        - This function assumes that each detection and ground truth file corresponds to one image and that the files are named such that detection and ground truth files match.
        - The function calculates the Intersection over Union (IoU) to determine matches between detection and ground truth bounding boxes.
          A detection is considered a true positive if it has an IoU of 0.5 or higher with any ground truth bounding box.
        - Ground truth bounding boxes that do not have any matching detection bounding box with an IoU of 0.5 or higher are considered false negatives.
        - Detections that do not match any ground truth bounding box with an IoU of 0.5 or higher are considered false positives.
    """
    # List and sort detection and ground truth files
    det_files = sorted([f for f in os.listdir(det_path) if f.endswith(".txt")])
    gt_files = sorted([f for f in os.listdir(gt_path) if f.endswith(".txt")])

    # Initialize counters for true positives (TP), false positives (FP), and false negatives (FN)
    tp, fp, fn = 0, 0, 0
    # Initialize lists to store detection scores and labels
    scores = []
    labels = []

    # Iterate over ground truth files
    for gt_file in gt_files:
        # Check if corresponding detection file exists
        if gt_file in det_files:
            # Read ground truth bounding boxes
            gt_bboxes = read_bboxes(
                os.path.join(gt_path, gt_file), format="ground_truth"
            )
            # Read detection bounding boxes with probabilities
            det_bboxes_with_probs = read_bboxes(
                os.path.join(det_path, gt_file), format="detection"
            )

            # Set to track matched ground truth bounding boxes
            matched_gt_idxs = set()

            # Iterate over detection bounding boxes
            for item in det_bboxes_with_probs:
                # Separate bounding box and probability score
                det_bbox, score = item if isinstance(item, tuple) else (item, 1.0)

                # Find best matching ground truth bounding box for each detection
                ious = [
                    (iou_aabb(det_bbox, gt_bbox), idx)
                    for idx, gt_bbox in enumerate(gt_bboxes)
                ]
                best_iou, best_idx = max(ious, key=lambda x: x[0], default=(0, -1))

                # Determine if detection is true positive
                is_tp = best_iou >= 0.5 and best_idx not in matched_gt_idxs
                # Append score and label
                scores.append(score)
                labels.append(is_tp)

                # Update counters based on true positive or false positive
                if is_tp:
                    tp += 1
                    matched_gt_idxs.add(best_idx)  # Mark this GT box as matched
                else:
                    fp += 1

            # Update calculation of false negatives based on unmatched GT boxes
            fn += len(gt_bboxes) - len(matched_gt_idxs)

    # Calculate precision, recall, and F1-score
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )

    # Generate ROC data and plot curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

    # Return precision, recall, and F1-score
    return precision, recall, f1_score
