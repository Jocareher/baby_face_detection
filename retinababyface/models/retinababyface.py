from typing import Optional, Dict,Tuple

import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter

from retinababyface.models.mobilenet import FPN, SSH

class OBBHead(nn.Module):
    """
    Head module for oriented bounding box (OBB) regression.
    """
    def __init__(self, inchannels: int = 64, num_anchors: int = 2):
        """
        Initializes the OBBHead module.

        Args:
            inchannels (int): Number of input channels. Defaults to 64.
            num_anchors (int): Number of anchors per location. Defaults to 2.
        """
        super().__init__()
        self.conv = nn.Conv2d(inchannels, num_anchors * 8, kernel_size=1)  # 1x1 convolution to predict OBB coordinates.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the OBBHead module.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Predicted OBB coordinates.
        """
        out = self.conv(x)  # Apply the convolution.
        out = out.permute(0, 2, 3, 1).contiguous()  # Rearrange the tensor dimensions.
        return out.view(out.shape[0], -1, 8)  # Reshape the tensor to (batch_size, num_anchors * H * W, 8).


class AngleHead(nn.Module):
    """
    Head module for angle prediction.
    """
    def __init__(self, inchannels: int = 64, num_anchors: int = 2):
        """
        Initializes the AngleHead module.

        Args:
            inchannels (int): Number of input channels. Defaults to 64.
            num_anchors (int): Number of anchors per location. Defaults to 2.
        """
        super().__init__()
        self.conv = nn.Conv2d(inchannels, num_anchors * 1, kernel_size=1)  # 1x1 convolution to predict angles.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AngleHead module.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Predicted angles.
        """
        out = self.conv(x)  # Apply the convolution.
        out = out.permute(0, 2, 3, 1).contiguous()  # Rearrange the tensor dimensions.
        return out.view(out.shape[0], -1, 1)  # Reshape the tensor to (batch_size, num_anchors * H * W, 1).


class ClassHead(nn.Module):
    """
    Head module for class prediction.
    """
    def __init__(self, inchannels: int = 64, num_classes: int = 6, num_anchors: int = 2):
        """
        Initializes the ClassHead module.

        Args:
            inchannels (int): Number of input channels. Defaults to 64.
            num_classes (int): Number of classes to predict. Defaults to 6.
            num_anchors (int): Number of anchors per location. Defaults to 2.
        """
        super().__init__()
        self.conv = nn.Conv2d(inchannels, num_anchors * num_classes, kernel_size=1)  # 1x1 convolution for class prediction.
        self.num_classes = num_classes  # Store the number of classes.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ClassHead module.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Predicted class logits.
        """
        out = self.conv(x)  # Apply the convolution.
        out = out.permute(0, 2, 3, 1).contiguous()  # Rearrange the tensor dimensions.
        return out.view(out.shape[0], -1, self.num_classes)  # Reshape the tensor.


class RetinaBabyFace(nn.Module):
    """
    RetinaBabyFace model with MobileNetV1 backbone and three heads:
    - Perspective classification
    - OBB regression
    - Angle prediction
    """
    def __init__(self, backbone: nn.Module, return_layers: Dict[str, str], in_channel: int, out_channel: int,
                 pretrain_path: Optional[str] = None, freeze_backbone: bool = True):
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
            checkpoint = torch.load(pretrain_path, map_location='cpu')  # Load the checkpoint.
            state_dict = checkpoint.get('state_dict', checkpoint)  # Get the state dictionary.

            # Extract only the backbone weights (filter "body.stage")
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('body.'):
                    k_clean = k.replace('body.', '')  # Remove the 'body.' prefix.
                    filtered_state_dict[k_clean] = v

            backbone.load_state_dict(filtered_state_dict, strict=False)  # Load the filtered weights.

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
        self.obb_head = nn.ModuleList([OBBHead(out_channel) for _ in range(3)])  # OBB regression heads.
        self.angle_head = nn.ModuleList([AngleHead(out_channel) for _ in range(3)])  # Angle prediction heads.
        self.class_head = nn.ModuleList([ClassHead(out_channel, num_classes=6) for _ in range(3)])  # Class prediction heads.

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the RetinaBabyFace model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Predicted class logits, OBB coordinates, and angles.
        """
        out = self.body(x)  # Extract features from the backbone.
        fpn_outs = self.fpn(out)  # Apply the FPN.
        features = [self.ssh1(fpn_outs[0]), self.ssh2(fpn_outs[1]), self.ssh3(fpn_outs[2])]  # Apply the SSH modules.

        obbs = torch.cat([self.obb_head[i](f) for i, f in enumerate(features)], dim=1)  # Concatenate OBB predictions.
        angles = torch.cat([self.angle_head[i](f) for i, f in enumerate(features)], dim=1)  # Concatenate angle predictions.
        persp_logits = torch.cat([self.class_head[i](f) for i, f in enumerate(features)], dim=1)  # Concatenate class predictions.

        return persp_logits, obbs, angles  # Return the predictions.