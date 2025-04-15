from typing import List, Tuple

import torch
import torch.nn as nn


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
        anchors_per_image = []  # List to store anchor boxes for each feature map level.

        for k, (fm_shape, stride) in enumerate(
            zip(self.fm_shapes, self.strides)
        ):  # Iterate through feature map levels.
            h, w = fm_shape  # Get the height and width of the feature map.
            scale = self.scales[k]  # Get the scale for the current level.
            anchors = []  # List to store anchors for the current level.

            for i in range(h):  # Iterate through rows of the feature map.
                for j in range(w):  # Iterate through columns of the feature map.
                    cx = (
                        j + 0.5
                    ) * stride  # Calculate the center x-coordinate of the anchor.
                    cy = (
                        i + 0.5
                    ) * stride  # Calculate the center y-coordinate of the anchor.

                    for ratio in self.ratios:  # Iterate through anchor ratios.
                        area = scale * scale  # Calculate the area of the anchor.
                        w_anchor = (
                            area / ratio
                        ) ** 0.5  # Calculate the width of the anchor.
                        h_anchor = (
                            w_anchor * ratio
                        )  # Calculate the height of the anchor.

                        x1 = (
                            cx - w_anchor / 2
                        )  # Calculate the x-coordinate of the top-left corner.
                        y1 = (
                            cy - h_anchor / 2
                        )  # Calculate the y-coordinate of the top-left corner.
                        x2 = (
                            cx + w_anchor / 2
                        )  # Calculate the x-coordinate of the top-right corner.
                        y2 = (
                            cy - h_anchor / 2
                        )  # Calculate the y-coordinate of the top-right corner.
                        x3 = (
                            cx + w_anchor / 2
                        )  # Calculate the x-coordinate of the bottom-right corner.
                        y3 = (
                            cy + h_anchor / 2
                        )  # Calculate the y-coordinate of the bottom-right corner.
                        x4 = (
                            cx - w_anchor / 2
                        )  # Calculate the x-coordinate of the bottom-left corner.
                        y4 = (
                            cy + h_anchor / 2
                        )  # Calculate the y-coordinate of the bottom-left corner.

                        anchor = [
                            x1,
                            y1,
                            x2,
                            y2,
                            x3,
                            y3,
                            x4,
                            y4,
                        ]  # Create the anchor box.
                        anchors.append(anchor)  # Add the anchor to the list.

            anchors = torch.tensor(
                anchors, dtype=torch.float32, device=device
            )  # Convert the anchors to a tensor.
            anchors_per_image.append(
                anchors
            )  # Add the anchors to the list of anchors per image.

        return torch.cat(
            anchors_per_image, dim=0
        )  # Concatenate the anchors from all levels.


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
