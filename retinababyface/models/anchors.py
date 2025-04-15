from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np


class AnchorGeneratorOBB:
    """
    Anchor generator tailored to the dataset's statistics.
    """

    def __init__(
        self,
        feature_map_shapes: List[Tuple[int, int]],
        strides: List[int],
        base_size: float,
        base_ratio: float,
        scale_factors: List[float] = [0.75, 1.0, 1.25],
        ratio_factors: List[float] = [0.85, 1.0, 1.15],
    ):
        """
        Initializes the AnchorGeneratorOBB module.

        Args:
            feature_map_shapes (List[Tuple[int, int]]): List of (height, width) tuples for each FPN level.
            strides (List[int]): List of strides corresponding to each FPN level.
            base_size (float): Average size of the bounding boxes (e.g., 209.5).
            base_ratio (float): Average height-to-width ratio (e.g., 1.18).
            scale_factors (List[float]): Multiplicative factors for the base size. Defaults to [0.75, 1.0, 1.25].
            ratio_factors (List[float]): Multiplicative factors for the base ratio. Defaults to [0.85, 1.0, 1.15].
        """
        self.fm_shapes = feature_map_shapes
        self.strides = strides
        self.scales = [
            base_size * s for s in scale_factors
        ]  # Calculate the anchor scales.
        self.ratios = [
            base_ratio * r for r in ratio_factors
        ]  # Calculate the anchor ratios.

    def generate_anchors(self, device: torch.device) -> torch.Tensor:
        """
        Generates anchor boxes for each feature map level.

        Args:
            device (torch.device): The device to generate anchors on (CPU or GPU).

        Returns:
            torch.Tensor: Tensor containing all generated anchor boxes (N_anchors_total, 8).
        """
        # Initialize an empty list to store anchors for each feature map level.
        anchors_per_image = []

        # Iterate through each feature map shape and corresponding stride.
        # For each feature map, generate anchors based on the center coordinates and the specified scales and ratios.
        for k, (fm_shape, stride) in enumerate(zip(self.fm_shapes, self.strides)):
            # Get the height and width of the feature map.
            h, w = fm_shape
            # Calculate the scale for the current feature map level.
            scale = self.scales[k]

            # Prepare all grid center coordinates at once
            grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            centers_x = (grid_x + 0.5) * stride  # shape (h, w)
            centers_y = (grid_y + 0.5) * stride  # shape (h, w)
            centers = np.stack([centers_x, centers_y], axis=-1).reshape(
                -1, 2
            )  # shape (h*w, 2)

            anchors = []

            # Generate anchors for each scale and ratio
            for ratio in self.ratios:
                # Calculate the width and height of the anchor box based on the scale and ratio.
                area = scale * scale
                # Calculate the width and height of the anchor box.
                w_anchor = np.sqrt(area / ratio)
                h_anchor = w_anchor * ratio

                # Calculate the half-width and half-height of the anchor box.
                dx = w_anchor / 2
                dy = h_anchor / 2

                # Generate anchor corners around each center
                corners = np.array(
                    [[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]]
                )  # shape (4, 2)

                # Calculate all anchor boxes by adding the corners to the centers
                all_anchors = (
                    centers[:, None, :] + corners[None, :, :]
                )  # shape (N, 4, 2)
                anchors.append(all_anchors.reshape(-1, 8))  # reshape to (N, 8)

            anchors = np.concatenate(anchors, axis=0)  # shape (N_total, 8)
            anchors_per_image.append(
                torch.tensor(anchors, dtype=torch.float32, device=device)
            )

        return torch.cat(anchors_per_image, dim=0)


def get_feature_map_shapes(
    model: nn.Module, input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640)
) -> List[Tuple[int, int]]:
    """
    Calculates the shape of the feature maps produced by the model's FPN.

    Args:
        model (nn.Module): The model to extract feature map shapes from.
        input_shape (Tuple[int, int, int, int]): The shape of the dummy input. Defaults to (1, 3, 640, 640).

    Returns:
        List[Tuple[int, int]]: List of (height, width) tuples for each feature map.
    """
    with torch.no_grad():  # Disable gradient calculation.
        dummy_input = torch.zeros(*input_shape).to(
            next(model.parameters()).device
        )  # Create a dummy input tensor.
        out = model.body(dummy_input)  # Pass the dummy input through the backbone.
        fpn_outs = model.fpn(out)  # Pass the backbone output through the FPN.
        return [
            (f.shape[2], f.shape[3]) for f in fpn_outs
        ]  # Return the shapes of the FPN outputs.
