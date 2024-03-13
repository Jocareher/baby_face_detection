import os

import numpy as np


def read_bboxes(file_path: str, format: str = "detection") -> list:
    """
    Reads bounding boxes from a text file and returns them as a list of lists.

    Args:
        - file_path (str): Path to the text file containing bounding box coordinates.
        - format (str): Format of the bounding boxes. It can be 'detection' or 'ground_truth'.

    Returns:
        - list: List of lists containing the bounding box coordinates.

    Description:
        This function reads the bounding box coordinates from a text file and returns them as a list of lists.
        If the format is 'ground_truth', it converts the coordinates from (x, y, w, h) format to (x1, y1, x2, y2) format to ensure consistency.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    if format == "ground_truth":
        # Converts ground truth from (x, y, w, h) to (x1, y1, x2, y2)
        bboxes = [list(map(float, line.strip().split())) for line in lines]
        bboxes = [[x, y, x + w, y + h] for x, y, w, h in bboxes]
    else:
        # Assumes detection format is already (x1, y1, x2, y2)
        bboxes = [list(map(float, line.strip().split())) for line in lines]
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


def compute_detection_metrics(det_path: str, gt_path: str) -> tuple:
    """
    Computes detection metrics (precision, recall, F1-score) for a set of detection results against ground truth annotations.

    This function calculates the following metrics:
        - Precision: The ratio of true positive detections to the total number of positive predictions made.
        It is calculated as TP / (TP + FP), where TP is the number of true positives (detections that correctly match a ground truth bounding box with IoU >= 0.5)
        and FP is the number of false positives (detections that do not correctly match any ground truth bounding box or match already detected ground truth bounding box).

        - Recall: The ratio of true positive detections to the total number of actual positives (ground truth bounding boxes).
        It is calculated as TP / (TP + FN), where FN is the number of false negatives (ground truth bounding boxes that were not detected by any detection bounding box).

        - F1-Score: The harmonic mean of precision and recall, providing a single metric to assess the balance between them.
        It is calculated as 2 * (precision * recall) / (precision + recall), giving equal weight to both precision and recall.

    Args:
        - det_path (str): Path to the directory containing detection bounding box files (.txt format), where each file contains bounding boxes for detections in an image.
        - gt_path (str): Path to the directory containing ground truth bounding box files (.txt format), where each file contains bounding boxes for actual objects in the corresponding image.

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

    # Iterate over ground truth files and corresponding detection files
    for gt_file in gt_files:
        gt_bboxes = read_bboxes(os.path.join(gt_path, gt_file), format="ground_truth")
        det_bboxes = (
            read_bboxes(os.path.join(det_path, gt_file)) if gt_file in det_files else []
        )

        # Track matched ground truth bounding boxes to avoid double counting
        matched = [False] * len(gt_bboxes)

        # Compare each detection bbox with each ground truth bbox
        for det_bbox in det_bboxes:
            iou_scores = [iou_aabb(det_bbox, gt_bbox) for gt_bbox in gt_bboxes]
            if iou_scores:
                best_iou_index = np.argmax(iou_scores)
                best_iou = iou_scores[best_iou_index]
                # If best IoU is above the threshold, count as TP or FP
                if best_iou >= 0.5:
                    if not matched[best_iou_index]:
                        tp += 1
                        matched[best_iou_index] = True
                    else:
                        fp += 1
                else:
                    fp += 1

        # Count unmatched ground truth bboxes as FN
        fn += len(gt_bboxes) - sum(matched)

    # Calculate precision, recall, and F1-score
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    )

    return precision, recall, f1_score
