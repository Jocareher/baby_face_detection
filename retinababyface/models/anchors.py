from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np


class AnchorGeneratorOBB:
    """
    AnchorGeneratorOBB generates a grid of oriented anchor boxes (OBBs) across feature map levels
    using dataset-specific statistics such as average box size and aspect ratio.

    Each anchor is represented by 4 corner points (flattened as 8 coordinates) in absolute pixel space,
    assuming axis-aligned rectangles (angle = 0). The anchors can later be refined by predicted deltas.

    This generator supports multiple scales and aspect ratios per location and works with multiple FPN levels.

    Attributes:
        fm_shapes (List[Tuple[int, int]]): Spatial dimensions (height, width) of each feature map.
        strides (List[int]): Stride (downsampling factor) associated with each feature map level.
        scales (List[float]): Absolute anchor sizes derived from base size and scale factors.
        ratios (List[float]): Aspect ratios derived from base ratio and ratio factors.
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
        Initializes the anchor generator with the given base statistics and FPN configuration.

        Args:
            feature_map_shapes (List[Tuple[int, int]]): (H, W) of each FPN level.
            strides (List[int]): Corresponding strides (e.g., [8, 16, 32]).
            base_size (float): Mean box size (√area) across dataset (e.g., 209.5).
            base_ratio (float): Mean aspect ratio (height / width) across dataset (e.g., 1.18).
            scale_factors (List[float]): Relative size multipliers (default: [0.75, 1.0, 1.25]).
            ratio_factors (List[float]): Aspect ratio multipliers (default: [0.85, 1.0, 1.15]).
        """
        self.fm_shapes = feature_map_shapes
        self.strides = strides
        self.scales = [base_size * s for s in scale_factors]
        self.ratios = [base_ratio * r for r in ratio_factors]

    def generate_anchors(self, device: torch.device) -> torch.Tensor:
        """
        Generates a complete set of anchor boxes for all FPN levels.

        Args:
            device (torch.device): The target device to place the generated anchors on.

        Returns:
            torch.Tensor: Tensor of shape (total_anchors, 8), each row representing
                          an anchor's 4 corner points as [x1, y1, x2, y2, x3, y3, x4, y4].
        """
        anchors_per_image = []

        # Iterate over each feature map shape and stride
        for fm_shape, stride in zip(self.fm_shapes, self.strides):
            #   Calculate the grid of centers for the current feature map
            h, w = fm_shape
            #   Create a grid of center points for the anchors
            grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            #  Convert grid coordinates to pixel space
            centers = np.stack(
                [(grid_x + 0.5) * stride, (grid_y + 0.5) * stride], axis=-1
            ).reshape(-1, 2)

            level_anchors = []
            # Iterate over each scale and aspect ratio
            for scale in self.scales:
                # Iterate over each aspect ratio
                for ratio in self.ratios:
                    # Calculate the width and height of the anchor box
                    area = scale**2
                    # Calculate the width and height of the anchor box
                    w_anchor = np.sqrt(area / ratio)
                    h_anchor = w_anchor * ratio
                    # Calculate the half-width and half-height
                    dx, dy = w_anchor / 2, h_anchor / 2

                    # Create axis-aligned rectangle corners centered at origin
                    corners = np.array(
                        [[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32
                    )  # (4, 2)

                    # Translate corners to all grid centers
                    anchors = centers[:, None, :] + corners[None, :, :]  # (N, 4, 2)
                    level_anchors.append(
                        anchors.reshape(-1, 8)
                    )  # Flattened OBBs (N, 8)
            # Concatenate all anchors for this feature map level
            all_level_anchors = np.concatenate(level_anchors, axis=0)
            anchors_per_image.append(
                torch.tensor(all_level_anchors, dtype=torch.float32, device=device)
            )

        return torch.cat(anchors_per_image, dim=0)


def get_feature_map_shapes(
    model: nn.Module, input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640)
) -> List[Tuple[int, int]]:
    """
    Calculates the shape of the feature maps produced by the model's FPN.
    """
    with torch.no_grad():
        dummy_input = torch.zeros(*input_shape).to(next(model.parameters()).device)
        # Extraemos con el backbone (en tu clase es .backbone, no .body)
        feats = model.backbone(dummy_input)  # dict: {'feat1':…, 'feat2':…, 'feat3':…}
        fpn_outs = model.fpn(feats)  # List[Tensor]
        return [(f.shape[2], f.shape[3]) for f in fpn_outs]
