import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.geometry import Polygon
from shapely.errors import TopologicalError

from retinababyface.loss.utils import match_anchors_to_targets


class RotationLoss(nn.Module):
    """
    Module for calculating the rotation loss between predicted and ground truth angles.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_angles: torch.Tensor,
        gt_angles: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calculates the rotation loss.

        The loss function is defined as:
        L_rot = 1 - cos(pred_angle - gt_angle)

        This loss penalizes the difference between predicted and ground truth angles,
        with a higher loss for larger differences.

        Args:
            pred_angles (torch.Tensor): Predicted angles tensor of shape (B, N, 1).
            gt_angles (torch.Tensor): Ground truth angles tensor of shape (B, N, 1).
            valid_mask (torch.Tensor, optional): Valid mask tensor of shape (B, N) to exclude certain predictions from loss calculation.

        Returns:
            torch.Tensor: The mean rotation loss.
        """
        # Both tensors are of shape (B, N, 1)
        loss = 1 - torch.cos(
            pred_angles.squeeze(-1) - gt_angles.squeeze(-1)
        )  # Calculate cosine difference between predicted and ground truth angles.
        if valid_mask is not None:
            loss = loss[
                valid_mask
            ]  # Apply valid mask to exclude certain predictions from loss.
        return loss.mean()  # Return the mean rotation loss.


class OBBLoss(nn.Module):
    """
    Module for calculating the Oriented Bounding Box (OBB) loss.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_obbs: torch.Tensor,
        gt_obbs: torch.Tensor,
        image_size: list,
        valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calculates the OBB loss.

        The loss function is defined as:
        L_obb = 1 - IoU(pred_obb, gt_obb)

        Where IoU is the Intersection over Union between the predicted and ground truth OBBs.
        This loss penalizes the difference in overlap between the predicted and ground truth OBBs.

        Args:
            pred_obbs (torch.Tensor): Predicted OBBs tensor of shape (B, N, 8) in normalized coordinates [0, 1].
            gt_obbs (torch.Tensor): Ground truth OBBs tensor of shape (B, N, 8) in normalized coordinates [0, 1].
            image_size (list): List of tuples (width, height) for each image in the batch, i.e., [(W1, H1), (W2, H2), ...].
            valid_mask (torch.Tensor, optional): Valid mask tensor of shape (B, N) to exclude certain predictions from loss calculation.

        Returns:
            torch.Tensor: The mean OBB loss.
        """
        B, N, _ = pred_obbs.shape  # Get batch size and number of OBBs.
        losses = []  # Initialize list to store individual OBB losses.

        for b in range(B):  # Iterate through each image in the batch.
            width, height = image_size[b]  # Get width and height of the current image.

            # Denormalize OBB coordinates.
            pred_boxes = pred_obbs[b] * torch.tensor(
                [width, height] * 4, device=pred_obbs.device
            )
            gt_boxes = gt_obbs[b] * torch.tensor(
                [width, height] * 4, device=gt_obbs.device
            )

            for i in range(N):  # Iterate through each OBB.
                if (
                    valid_mask is not None and not valid_mask[b, i]
                ):  # Skip OBB if it's masked out.
                    continue

                try:
                    pred_poly = Polygon(
                        pred_boxes[i].view(4, 2).detach().cpu().numpy()
                    )  # Create Shapely Polygon from predicted OBB.
                    gt_poly = Polygon(
                        gt_boxes[i].view(4, 2).detach().cpu().numpy()
                    )  # Create Shapely Polygon from ground truth OBB.

                    if (
                        not pred_poly.is_valid or not gt_poly.is_valid
                    ):  # Skip OBB if either polygon is invalid.
                        continue

                    inter = pred_poly.intersection(
                        gt_poly
                    ).area  # Calculate intersection area between predicted and ground truth OBBs.
                    union = pred_poly.union(
                        gt_poly
                    ).area  # Calculate union area between predicted and ground truth OBBs.

                    iou = inter / union if union > 0 else 0.0  # Calculate IoU.
                    loss_i = 1.0 - iou  # Calculate OBB loss as 1 - IoU.
                    losses.append(loss_i)  # Append OBB loss to the list.

                except (
                    ValueError,
                    TopologicalError,
                ):  # Handle potential errors during Polygon operations.
                    continue

        if len(losses) == 0:  # Return 0 loss if no valid OBB losses were calculated.
            return torch.tensor(0.0, requires_grad=True).to(pred_obbs.device)

        return torch.tensor(
            losses, device=pred_obbs.device
        ).mean()  # Return the mean OBB loss.


class MultiTaskLoss(nn.Module):
    """
    Module for calculating the multi-task loss for perspective classification, OBB regression, and rotation prediction.
    """

    def __init__(
        self,
        lambda_class: float = 1.0,
        lambda_obb: float = 1.0,
        lambda_rot: float = 1.0,
    ):
        super().__init__()
        self.class_loss_fn = (
            nn.CrossEntropyLoss()
        )  # CrossEntropyLoss for perspective classification.
        self.obb_loss_fn = OBBLoss()  # OBBLoss for OBB regression.
        self.rotation_loss_fn = RotationLoss()  # RotationLoss for angle prediction.
        self.lambda_class = lambda_class  # Weight for perspective classification loss.
        self.lambda_obb = lambda_obb  # Weight for OBB regression loss.
        self.lambda_rot = lambda_rot  # Weight for angle prediction loss.

    def forward(
        self, pred: tuple, targets: dict, anchors: torch.Tensor, image_sizes: list
    ) -> tuple:
        """
        Calculates the multi-task loss.

        The total loss is defined as:
        L_total = lambda_class * L_class + lambda_obb * L_obb + lambda_rot * L_rot

        Where:
        - L_class is the CrossEntropyLoss for perspective classification.
        - L_obb is the OBBLoss for OBB regression.
        - L_rot is the RotationLoss for angle prediction.
        - lambda_class, lambda_obb, and lambda_rot are weights for each loss component.

        Args:
            pred (tuple): Tuple containing predicted perspective logits, OBBs, and angles.
            targets (dict): Dictionary containing ground truth boxes, angles, and classes.
            anchors (torch.Tensor): Tensor of anchor boxes.
            image_sizes (list): List of tuples (width, height) for each image in the batch.

        Returns:
            tuple: Total loss, perspective loss, OBB loss, and angle loss.
        """
        (
            persp_logits,
            obbs_pred,
            angles_pred,
        ) = pred  # (B, N, 6), (B, N, 8), (B, N, 1)  # Unpack predictions.
        boxes_gt = targets["boxes"]  # Ground truth OBBs.
        angles_gt = targets["angle"]  # Ground truth angles.
        classes_gt = targets["class_idx"]  # Ground truth classes.

        (
            matched_idx,
            matched_classes,
            matched_boxes,
            matched_angles,
        ) = match_anchors_to_targets(
            anchors, boxes_gt, classes_gt, angles_gt
        )  # Match anchors to ground truth targets.

        losses_class, losses_obb, losses_angle = (
            [],
            [],
            [],
        )  # Initialize lists to store individual losses.

        for b in range(len(matched_idx)):  # Iterate through each image in the batch.
            idx = matched_idx[b]  # Get indices of positive anchors.
            if len(idx) == 0:  # Skip image if no positive anchors were found.
                continue
            pred_logits = persp_logits[b][
                idx
            ]  # (P, 6)  # Get predicted logits for positive anchors.
            pred_obbs = obbs_pred[b][
                idx
            ]  # (P, 8)  # Get predicted OBBs for positive anchors.
            pred_angles = angles_pred[b][
                idx
            ]  # (P, 1)  # Get predicted angles for positive anchors.
            gt_classes = matched_classes[
                b
            ]  # (P,)  # Get ground truth classes for positive anchors.
            gt_obbs = matched_boxes[
                b
            ]  # (P, 8)  # Get ground truth OBBs for positive anchors.
            gt_angles = matched_angles[
                b
            ]  # Get ground truth angles for positive anchors.
            if gt_angles.ndim == 1:
                gt_angles = gt_angles.unsqueeze(
                    -1
                )  # Ensure gt_angles has shape (P, 1).

            pred_angles = torch.remainder(
                pred_angles, 2 * math.pi
            )  # Normalize predicted angles to [0, 2π).
            gt_angles = torch.remainder(
                gt_angles, 2 * math.pi
            )  # Normalize ground truth angles to [0, 2π).

            # Get current image size.
            W, H = image_sizes[b]
            valid_mask = torch.ones(
                gt_obbs.shape[0], dtype=torch.bool, device=gt_obbs.device
            )  # Create valid mask for OBB loss.

            losses_class.append(
                self.class_loss_fn(pred_logits, gt_classes)
            )  # Calculate perspective classification loss.
            losses_obb.append(
                self.obb_loss_fn(
                    pred_obbs.unsqueeze(0),
                    gt_obbs.unsqueeze(0),
                    image_size=[(W, H)],
                    valid_mask=valid_mask.unsqueeze(0),
                )
            )  # Calculate OBB loss.
            losses_angle.append(
                F.smooth_l1_loss(torch.cos(pred_angles), torch.cos(gt_angles))
            )  # Calculate angle prediction loss.

        if len(losses_class) == 0:  # Return 0 loss if no positive anchors were found.
            return (
                torch.tensor(0.0, requires_grad=True).to(persp_logits.device),
                0.0,
                0.0,
                0.0,
            )

        loss_persp = torch.stack(
            losses_class
        ).mean()  # Calculate mean perspective classification loss.
        loss_vertex = torch.stack(losses_obb).mean()  # Calculate mean OBB loss.
        loss_angle = torch.stack(
            losses_angle
        ).mean()  # Calculate mean angle prediction loss.

        total_loss = (
            self.lambda_class * loss_persp
            + self.lambda_obb * loss_vertex
            + self.lambda_rot * loss_angle
        )  # Calculate total multi-task loss.

        return (
            total_loss,
            loss_persp.item(),
            loss_vertex.item(),
            loss_angle.item(),
        )  # Return total loss and individual losses.
