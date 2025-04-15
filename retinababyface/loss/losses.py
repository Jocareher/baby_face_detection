import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.geometry import Polygon
from shapely.errors import TopologicalError

from .utils import match_anchors_to_targets


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
        # Unpack the predicted values.
        persp_logits, obbs_pred, angles_pred = pred  # (B, N, 6), (B, N, 8), (B, N, 1)
        # Unpack the ground truth values.
        boxes_gt = targets["boxes"]
        # Normalize the ground truth angles to be within [0, 2*pi).
        angles_gt = torch.remainder(
            targets["angle"], 2 * math.pi
        )  # Normalize all GT angles
        classes_gt = targets["class_idx"]

        # Match anchors to targets.
        # The function returns the indices of matched anchors, classes, boxes, and angles.
        (
            matched_idx,
            matched_classes,
            matched_boxes,
            matched_angles,
        ) = match_anchors_to_targets(anchors, boxes_gt, classes_gt, angles_gt)
        # Normalize the predicted angles to be within [0, 2*pi).
        angles_pred = torch.remainder(
            angles_pred, 2 * math.pi
        )  # Normalize all predicted angles

        losses_class, losses_obb, losses_angle = [], [], []

        for b in range(len(matched_idx)):
            # Get the indices of matched anchors for the current image.
            idx = matched_idx[b]
            if len(idx) == 0:
                continue

            # Get the predicted values for the matched anchors.
            pred_logits = persp_logits[b][idx]
            # Get the predicted OBBs and angles for the matched anchors.
            pred_obbs = obbs_pred[b][idx]
            # Get the predicted angles for the matched anchors.
            pred_angles = angles_pred[b][idx]

            # Get the ground truth values for the matched anchors.
            gt_classes = matched_classes[b]
            # Get the ground truth OBBs and angles for the matched anchors.
            gt_obbs = matched_boxes[b]
            # Get the ground truth angles for the matched anchors.
            gt_angles = matched_angles[b]

            if gt_angles.ndim == 1:
                # Add a dimension to gt_angles to match the predicted angles.
                gt_angles = gt_angles.unsqueeze(-1)

            # Get the image size for the current image.
            W, H = image_sizes[b]
            # Create a valid mask for the ground truth OBBs.
            # The mask is used to exclude invalid OBBs from the loss calculation.
            # In this case, all OBBs are considered valid.
            # valid_mask = torch.ones(gt_obbs.shape[0], dtype=torch.bool, device=gt_obbs.device)
            valid_mask = torch.ones(
                gt_obbs.shape[0], dtype=torch.bool, device=gt_obbs.device
            )
            # valid_mask = valid_mask & (gt_obbs[:, 0] > 0) & (gt_obbs[:, 1] > 0) & (gt_obbs[:, 2] > 0) & (gt_obbs[:, 3] > 0)
            losses_class.append(self.class_loss_fn(pred_logits, gt_classes))
            losses_obb.append(
                self.obb_loss_fn(
                    pred_obbs.unsqueeze(0),
                    gt_obbs.unsqueeze(0),
                    image_size=[(W, H)],
                    valid_mask=valid_mask.unsqueeze(0),
                )
            )
            losses_angle.append(
                self.rotation_loss_fn(pred_angles, gt_angles, valid_mask)
            )

        if len(losses_class) == 0:
            # If no valid losses were calculated, return zero losses.
            return (
                torch.tensor(0.0, requires_grad=True).to(persp_logits.device),
                0.0,
                0.0,
                0.0,
            )
        # Calculate the mean loss for each component.
        loss_persp = torch.stack(losses_class).mean()
        loss_vertex = torch.stack(losses_obb).mean()
        loss_angle = torch.stack(losses_angle).mean()
        # Calculate the total loss as a weighted sum of the individual losses.
        total_loss = (
            self.lambda_class * loss_persp
            + self.lambda_obb * loss_vertex
            + self.lambda_rot * loss_angle
        )
        ## Return the total loss and individual losses.
        return total_loss, loss_persp.item(), loss_vertex.item(), loss_angle.item()
