from typing import List, Tuple
import math

import torch
import torch.nn as nn
import numpy as np


import torch
import torch.nn as nn
import math


class AnchorGeneratorOBB(nn.Module):
    """
    Generates a dense set of oriented bounding box (OBB) anchors for multiple feature levels
    (e.g., FPN stages), using predefined width-height ratios, scales, and rotation angles.

    Each anchor is represented by its 4 corners in absolute pixel coordinates, flattened as:
        [x1, y1, x2, y2, x3, y3, x4, y4]

    This generator:
    - Vectorizes the creation of anchors over all combinations of (scale, ratio, angle).
    - Precomputes anchor corner offsets centered at the origin and caches them as a buffer.
    - Efficiently shifts these anchors to every grid cell center on each feature map level.

    Args:
        base_size (float): Base size (area) used to compute anchor width and height.
        base_ratio (float): Base aspect ratio (height/width).
        scale_factors (list[float]): Multipliers for scaling the base size.
        ratio_factors (list[float]): Multipliers for adjusting the base aspect ratio.
        angles (list[float]): Rotation angles (in radians) applied to each anchor.

    Attributes:
        corner_offsets (torch.Tensor): Cached (A, 8) tensor of pre-rotated anchor templates.
        A (int): Total number of anchor templates per spatial location (len(scale_ratio × angles)).
    """

    def __init__(
        self,
        base_size: float,
        base_ratio: float,
        scale_factors: list[float] = None,
        ratio_factors: list[float] = None,
        angles: list[float] = None,
    ):
        super().__init__()
        # Compute (w, h) pairs for each scale and ratio combination
        wh_list = []
        for s in scale_factors:
            # Compute width and height for each scale and ratio
            # The base size is the area of the anchor, so we need to compute width and height
            # based on the base size and the aspect ratio.
            # The area of the anchor is given by base_size, and the aspect ratio is given by base_ratio.
            # The width and height are computed as follows:
            size = base_size * s
            for rf in ratio_factors:
                ratio = base_ratio * rf
                w = math.sqrt(size**2 / ratio)
                h = w * ratio
                wh_list.append((w, h))
        wh = torch.tensor(wh_list, dtype=torch.float32)

        # For each (w, h) and rotation angle, compute the rotated anchor corners
        offsets = []
        for w, h in wh:
            # Compute the 4 corners of the anchor rectangle
            # The rectangle is centered at the origin, so we need to compute the offsets
            # based on the width and height of the rectangle.
            # The corners are computed as follows:
            # (x1, y1) = (-w/2, -h/2)
            # (x2, y2) = (w/2, -h/2)
            # (x3, y3) = (w/2, h/2)
            # (x4, y4) = (-w/2, h/2)
            # The corners are stored in a tensor of shape (4, 2), where each row is a corner
            # and each column is the x and y coordinate of the corner.
            # The corners are then reshaped to a tensor of shape (8,) by flattening the 4x2 tensor.
            # The offsets are computed as follows:
            # (x1, y1, x2, y2, x3, y3, x4, y4) = (-w/2, -h/2, w/2, -h/2, w/2, h/2, -w/2, h/2)
            # The offsets are stored in a tensor of shape (8,) for each (w, h) pair.
            # The offsets are then rotated by the angles specified in the angles list.
            dx, dy = w / 2, h / 2
            rect = torch.tensor(
                [[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=torch.float32
            )  # (4, 2)
            for theta in angles:
                c, s = math.cos(theta), math.sin(theta)
                R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)  # (2, 2)
                rot = rect @ R.T
                offsets.append(rot.reshape(-1))  # (8,)
        offsets = torch.stack(offsets, dim=0)  # (A, 8)

        self.register_buffer(
            "corner_offsets", offsets
        )  # Will be moved with model.to(...)
        self.A = offsets.size(0)

    def generate_anchors(
        self,
        feature_map_shapes: list[tuple[int, int]],
        strides: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generates anchors for all FPN levels given their shapes and strides.

        For each feature map, anchors are placed densely at every location (grid center)
        and replicated across all precomputed anchor templates (A per location).

        Args:
            feature_map_shapes (list[tuple[int, int]]): Feature map shapes as (H, W) for each level.
            strides (list[int]): Downsampling stride of each feature map relative to the input.
            device (torch.device): Target device to allocate the anchors.

        Returns:
            torch.Tensor: Tensor of shape (N_total, 8), where N_total = sum(H_i * W_i * A),
                          with each row being a rotated anchor in (x1, y1, ..., x4, y4) format.
        """
        all_anc = []
        offs = self.corner_offsets.to(device).view(1, self.A, 4, 2)  # (1, A, 4, 2)

        for (h, w), stride in zip(feature_map_shapes, strides):
            # Compute center coordinates of each grid cell
            xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
            ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
            xv, yv = torch.meshgrid(xs, ys, indexing="xy")
            centres = torch.stack([xv.reshape(-1), yv.reshape(-1)], dim=1)  # (H*W, 2)

            # Add the precomputed corner offsets to each center
            C = centres.size(0)
            ctr = centres.view(C, 1, 1, 2)  # (C, 1, 1, 2)
            corners = ctr + offs  # (C, A, 4, 2)
            anchors = corners.view(-1, 8)  # (C * A, 8)

            all_anc.append(anchors)

        return torch.cat(all_anc, dim=0)  # (N_total, 8)


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
