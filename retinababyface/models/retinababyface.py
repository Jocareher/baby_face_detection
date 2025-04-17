from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter

from .mobilenet import FPN, SSH


class OBBHead(nn.Module):
    """
    Head module for predicting the 8 normalized offsets (Δx, Δy) of the 4 vertices
    of an oriented bounding box (OBB) relative to its anchor.

    The output is constrained to the range [-1, 1] using a tanh activation, so that
    vertex displacements remain within a reasonable distance from the anchor.

    Output shape:
        - Input: (B, C, H, W)
        - Output: (B, N, 8) where N = H × W × num_anchors
    """

    def __init__(self, in_ch: int, num_anchors: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_anchors * 8, kernel_size=1)

    def forward(self, x):
        # Apply 1x1 convolution, reshape and apply tanh to constrain output to [-1, 1]
        # The output shape is (B, num_anchors * H * W, 8)
        # The 8 values correspond to the 4 vertices of the OBB.
        # The vertices are represented as (Δx1, Δy1, Δx2, Δy2, Δx3, Δy3, Δx4, Δy4)
        # The output is reshaped to (B, N, 8) where N = H × W × num_anchors
        return torch.tanh(self.conv(x).permute(0, 2, 3, 1).contiguous()).view(
            x.size(0), -1, 8
        )


class AngleHead(nn.Module):
    """
    Head module for predicting the rotation angle (in radians) of each oriented bounding box (OBB).

    The predicted angle is constrained to the range [0, 2π] using a sigmoid activation followed by scaling.

    Output shape:
        - Input: (B, C, H, W)
        - Output: (B, N, 1) where N = H × W × num_anchors
    """

    def __init__(self, inchannels: int = 64, num_anchors: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(inchannels, num_anchors * 1, kernel_size=1)

    def forward(self, x):
        # Apply 1x1 convolution, reshape and scale sigmoid output to [0, 2π]
        # The output shape is (B, num_anchors * H * W, 1)
        # The 1 value corresponds to the rotation angle of the OBB.
        # The output is reshaped to (B, N, 1) where N = H × W × num_anchors
        # The angle is represented in radians.
        return (
            torch.sigmoid(self.conv(x).permute(0, 2, 3, 1).contiguous()) * 2 * math.pi
        ).view(x.size(0), -1, 1)


class ClassHead(nn.Module):
    """
    Head module for class prediction.
    """

    def __init__(
        self, inchannels: int = 64, num_classes: int = 6, num_anchors: int = 2
    ):
        """
        Initializes the ClassHead module.

        Args:
            inchannels (int): Number of input channels. Defaults to 64.
            num_classes (int): Number of classes to predict. Defaults to 6.
            num_anchors (int): Number of anchors per location. Defaults to 2.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            inchannels, num_anchors * num_classes, kernel_size=1
        )  # 1x1 convolution for class prediction.
        self.num_classes = num_classes  # Store the number of classes.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ClassHead module.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Predicted class logits.
        """
        # Apply the convolution and rearrange the tensor dimensions.
        # The output shape is (batch_size, num_anchors * H * W, num_classes).
        # The num_classes values correspond to the class logits for each anchor.
        # The logits are not normalized, so they can be used directly for classification.
        return (
            self.conv(x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(x.size(0), -1, self.num_classes)
        )


class RetinaBabyFace(nn.Module):
    """
    RetinaBabyFace model with MobileNetV1 backbone and three heads:
    - Class classification
    - OBB regression
    - Angle prediction
    """

    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        out_channel: int,
        pretrain_path: Optional[str] = None,
        freeze_backbone: bool = True,
    ):
        """
        Initializes the RetinaBabyFace model.

        Args:
            backbone (nn.Module): Backbone network (e.g., MobileNetV1).
            return_layers (Dict[str, str]): Layers to return from the backbone.
            in_channel (int): Number of input channels to the FPN.
            out_channel (int): Number of output channels from the FPN.
            pretrain_path (Optional[str]): Path to pretrained weights. Defaults to None.
            freeze_backbone (bool): Whether to freeze the backbone weights. Defaults to True.
        """
        super().__init__()

        # Load pretrained weights BEFORE IntermediateLayerGetter
        if pretrain_path:
            checkpoint = torch.load(
                pretrain_path, map_location="cpu"
            )  # Load the checkpoint.
            state_dict = checkpoint.get(
                "state_dict", checkpoint
            )  # Get the state dictionary.

            # Extract only the backbone weights (filter "body.stage")
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("body."):
                    k_clean = k.replace("body.", "")  # Remove the 'body.' prefix.
                    filtered_state_dict[k_clean] = v

            backbone.load_state_dict(
                filtered_state_dict, strict=False
            )  # Load the filtered weights.

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False  # Freeze the backbone weights.

        # Backbone feature extractor
        self.body = IntermediateLayerGetter(backbone, return_layers)

        # FPN + SSH
        in_channels_list = [64, 128, 256]  # Input channels for the FPN.
        self.fpn = FPN(in_channels_list, out_channel)  # Feature Pyramid Network.
        self.ssh1 = SSH(out_channel, out_channel)  # SSH module 1.
        self.ssh2 = SSH(out_channel, out_channel)  # SSH module 2.
        self.ssh3 = SSH(out_channel, out_channel)  # SSH module 3.

        # Heads
        self.obb_head = nn.ModuleList(
            [OBBHead(out_channel, num_anchors=9) for _ in range(3)]
        )  # OBB regression heads.
        self.angle_head = nn.ModuleList(
            [AngleHead(out_channel, num_anchors=9) for _ in range(3)]
        )  # Angle prediction heads.
        self.class_head = nn.ModuleList(
            [ClassHead(out_channel, num_anchors=9, num_classes=5) for _ in range(3)]
        )  # Class prediction heads.

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the RetinaBabyFace model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Predicted class logits, OBB coordinates, and angles.
        """
        out = self.body(x)  # Extract features from the backbone
        fpn_out = self.fpn(out)  # Apply FPN (returns List[Tensor])

        # Apply SSH modules
        features = [self.ssh1(fpn_out[0]), self.ssh2(fpn_out[1]), self.ssh3(fpn_out[2])]

        # Run heads and concatenate outputs
        # Each head processes the corresponding feature map.
        # The outputs are concatenated along the channel dimension.
        # The final output shapes are:
        # - persp_logits: (batch_size, num_anchors * H * W, num_classes)
        # - obbs: (batch_size, num_anchors * H * W, 8)
        # - angles: (batch_size, num_anchors * H * W, 1)
        persp_logits = torch.cat(
            [head(f) for head, f in zip(self.class_head, features)], dim=1
        )
        obbs = torch.cat([head(f) for head, f in zip(self.obb_head, features)], dim=1)
        angles = torch.cat(
            [head(f) for head, f in zip(self.angle_head, features)], dim=1
        )

        return persp_logits, obbs, angles
