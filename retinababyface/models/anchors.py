from typing import List, Tuple
import math

import torch
import torch.nn as nn
import numpy as np


class AnchorGeneratorOBB:
    """
    Generate a dense set of oriented anchor boxes (OBBs) for several FPN levels.

    Each anchor is stored as *absolute* pixel corner coordinates
    (x1, y1, x2, y2, x3, y3, x4, y4).

    ── Parameters ─────────────────────────────────────────────────────────────
    feature_map_shapes : list[(h, w)]
        Spatial size produced by the FPN at every level.
    strides            : list[int]
        Down‑sampling factor of each level (image_px / feature_px).
    base_size          : float
        Average square‑root area of objects in the dataset (≈ sqrt(w*h)).
    base_ratio         : float
        Average aspect ratio (height / width) in the dataset.
    scale_factors      : list[float], default (0.5, 0.75, 1.0, 1.5)
        Multipliers applied to *base_size* (controls absolute scale).
    ratio_factors      : list[float], default (0.85, 1.0, 1.15)
        Multipliers applied to *base_ratio* (controls width‑to‑height).
    angles             : list[float], default (-π/4, -π/8, 0, π/8)
        In‑plane rotations **in radians** that will be added for *every*
        scale×ratio combination.  Extend this list if your task needs
        more orientations (e.g. 8 or 12 angles evenly spaced in 180°).
    """

    def __init__(
        self,
        feature_map_shapes: List[Tuple[int, int]],
        strides: List[int],
        base_size: float,
        base_ratio: float,
        scale_factors: List[float] = None,
        ratio_factors: List[float] = None,
        angles: List[float] = None,
    ) -> None:
        self.fm_shapes = feature_map_shapes
        self.strides = strides

        scale_factors = scale_factors or [0.5, 0.75, 1.0, 1.5]
        ratio_factors = ratio_factors or [0.85, 1.0, 1.15]
        angles = angles or [-math.pi / 4, -math.pi / 8, 0.0, math.pi / 8]

        # Pre‑compute absolute scales, aspect ratios and rotations
        self.scales = [base_size * s for s in scale_factors]
        self.ratios = [base_ratio * r for r in ratio_factors]
        self.angles = angles

    # --------------------------------------------------------------------- #
    #                             Public method                              #
    # --------------------------------------------------------------------- #
    def generate_anchors(self, device: torch.device) -> torch.Tensor:
        """
        Return
        ------
        anchors_xy : torch.Tensor, shape (N_total, 8), dtype=float32
            Flattened 4‑point representation in the *same device*
            requested by the caller.
        """
        all_anchors: list[torch.Tensor] = []

        for (h, w), stride in zip(self.fm_shapes, self.strides):
            # 1)  Grid of centres in image coordinates (pixel domain)
            grid_y, grid_x = np.meshgrid(
                np.arange(h, dtype=np.float32),
                np.arange(w, dtype=np.float32),
                indexing="ij",
            )
            centres = np.stack(
                [(grid_x + 0.5) * stride, (grid_y + 0.5) * stride], axis=-1
            ).reshape(
                -1, 2
            )  # (H*W, 2)

            level_anchors: list[np.ndarray] = []

            # 2)  Enumerate scale × ratio × angle
            for scale in self.scales:
                for ratio in self.ratios:
                    area = scale**2
                    width = math.sqrt(area / ratio)
                    height = width * ratio

                    # axis‑aligned rectangle centred at the origin
                    dx, dy = width / 2, height / 2
                    rect = np.array(
                        [[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]],
                        dtype=np.float32,
                    )  # (4, 2)

                    for theta in self.angles:
                        c, s = math.cos(theta), math.sin(theta)
                        rot = np.array([[c, -s], [s, c]], dtype=np.float32)

                        rotated = rect @ rot.T  # (4, 2)
                        anchors_xy = (
                            centres[:, None, :] + rotated[None, :, :]
                        ).reshape(
                            -1, 8
                        )  # (num_centres, 8)

                        level_anchors.append(anchors_xy)

            # 3)  Concatenate all combinations for this level
            all_level = np.concatenate(level_anchors, axis=0)
            all_anchors.append(
                torch.as_tensor(all_level, dtype=torch.float32, device=device)
            )

        # 4)  Merge all levels
        return torch.cat(all_anchors, dim=0)


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
