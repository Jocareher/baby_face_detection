from typing import Optional, Dict, Tuple
import math

import torch
import torch.nn as nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    vgg16,
    VGG16_Weights,
    densenet121,
    DenseNet121_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)
from torchvision.models.feature_extraction import create_feature_extractor

from .mobilenet import FPN, SSH, MobileNetV1


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
    RetinaBabyFace model integrating backbone, FPN, SSH blocks,
    and multiple prediction heads for oriented bounding box detection, angle estimation, and class prediction.
    """

    def __init__(
        self,
        backbone_name: str = "mobilenetv1",
        out_channel: int = 64,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        """
        Initializes the RetinaBabyFace model.

        Args:
            backbone_name (str): Name of the backbone to use (e.g., "mobilenetv1", "resnet50", "vgg16").
            out_channel (int): Number of output channels for FPN layers.
            pretrained (bool): Whether to load pretrained weights for the backbone.
            freeze_backbone (bool): Whether to freeze backbone weights during training.
        """
        super().__init__()

        # Build backbone and retrieve feature extractor, return layers, and in_channels_list
        self.backbone, return_layers, in_channels_list = self.make_backbone(
            backbone_name, pretrained
        )

        # Feature Pyramid Network
        self.fpn = FPN(in_channels_list, out_channel)

        # SSH layers applied on each FPN output
        self.ssh1 = SSH(out_channel, out_channel)
        self.ssh2 = SSH(out_channel, out_channel)
        self.ssh3 = SSH(out_channel, out_channel)

        # Prediction heads: Oriented bounding boxes, rotation angles, and class logits
        self.obb_head = nn.ModuleList(
            [OBBHead(out_channel, num_anchors=9) for _ in range(3)]
        )
        self.angle_head = nn.ModuleList(
            [AngleHead(out_channel, num_anchors=9) for _ in range(3)]
        )
        self.class_head = nn.ModuleList(
            [ClassHead(out_channel, num_anchors=9, num_classes=6) for _ in range(3)]
        )

        # Optionally freeze backbone parameters
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

            # Set the backbone to evaluation mode
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def make_backbone(
        self, name: str, pretrained: bool
    ) -> tuple[nn.Module, dict, list[int]]:
        """
        Creates and returns a feature extractor from a specified backbone.

        Args:
            name (str): Name of the backbone model.
            pretrained (bool): Whether to use pretrained weights.

        Returns:
            Tuple containing:
                - feature extractor (nn.Module)
                - return_layers (dict): Mapping of layer names to output names
                - in_channels_list (list[int]): Channels for each returned feature map
        """
        if name == "resnet50":
            # Using torchvision's ResNet50 as backbone
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            # Load the ResNet50 model with pretrained weights
            model = resnet50(weights=weights)
            # Extract the feature extractor from the model
            return_layers = {"layer2": "feat1", "layer3": "feat2", "layer4": "feat3"}
            # Define the channels for each returned feature map
            in_channels_list = [512, 1024, 2048]
            # Create the feature extractor
            # The return layers are the last layers of each block in ResNet50
            # The output names are "feat1", "feat2", and "feat3"
            # corresponding to the output of the last conv layer in each block
            feat_ext = create_feature_extractor(model, return_layers)

        elif name == "vgg16":
            # Using torchvision's VGG16 as backbone
            weights = VGG16_Weights.DEFAULT if pretrained else None
            # Load the VGG16 model with pretrained weights
            model = vgg16(weights=weights).features
            # Extract the feature extractor from the model
            # The return layers are the convolutional layers of the model
            # The output names are "feat1", "feat2", and "feat3"
            # corresponding to the output of the last conv layer in each block
            return_layers = {"16": "feat1", "23": "feat2", "30": "feat3"}
            # Define the channels for each returned feature map
            in_channels_list = [256, 512, 512]
            # Create the feature extractor
            feat_ext = create_feature_extractor(model, return_layers)

        elif name == "densenet121":
            # Using torchvision's DenseNet121 as backbone
            weights = DenseNet121_Weights.DEFAULT if pretrained else None
            # Load the DenseNet121 model with pretrained weights
            model = densenet121(weights=weights).features
            # Extract the feature extractor from the model
            # The return layers are the dense blocks of the model
            # The output names are "feat1", "feat2", and "feat3"
            # corresponding to the output of the dense blocks
            # The return_layers dictionary maps the dense block names to output names
            return_layers = {
                "denseblock2": "feat1",
                "denseblock3": "feat2",
                "denseblock4": "feat3",
            }
            # Define the channels for each returned feature map
            in_channels_list = [512, 1024, 1024]
            feat_ext = create_feature_extractor(model, return_layers)

        elif name == "vit":
            # Using torchvision's Vision Transformer (ViT) as backbone
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            # Load the ViT model with pretrained weights
            vit = vit_b_16(weights=weights)
            # Extract the feature extractor from the model
            # The return layers are the transformer encoder layers of the model
            # The output names are "feat1", "feat2", and "feat3"
            # corresponding to the output of the last transformer encoder layers
            return_layers = {
                "encoder.layers.encoder_layer_2": "feat1",
                "encoder.layers.encoder_layer_5": "feat2",
                "encoder.layers.encoder_layer_8": "feat3",
            }
            # Define the channels for each returned feature map
            in_channels_list = [768, 768, 768]
            # Create the feature extractor
            # The return layers are the last layers of each block in ViT
            feat_seq = create_feature_extractor(vit, return_layers)
            # Convert the sequence output to 2D feature maps
            # The output names are "feat1", "feat2", and "feat3"
            feat_ext = ViTFeature2D(feat_seq, patch_size=16)

        else:
            #
            model = MobileNetV1()
            # Using MobileNetV1 as backbone
            # Extract the feature extractor from the model
            # The return layers are the last layers of each block in MobileNetV1
            return_layers = {"stage1": "feat1", "stage2": "feat2", "stage3": "feat3"}
            in_channels_list = [64, 128, 256]
            feat_ext = create_feature_extractor(model, return_layers)

        return feat_ext, return_layers, in_channels_list

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the RetinaBabyFace model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tuple containing:
                - logits (torch.Tensor): Class logits, shape (B, N, num_classes)
                - obbs (torch.Tensor): Predicted OBB vertex displacements, shape (B, N, 8)
                - angs (torch.Tensor): Predicted angles in radians, shape (B, N, 1)
        """
        # Extract features from the backbone
        # The input shape is (B, C, H, W)
        # The output shape is (B, C, H, W) for each feature level
        feats = self.backbone(x)
        # Apply the FPN to the extracted features
        p3, p4, p5 = self.fpn(feats)
        # Apply the SSH blocks to the FPN outputs
        f1, f2, f3 = self.ssh1(p3), self.ssh2(p4), self.ssh3(p5)

        # Concatenate the outputs from the SSH blocks
        logits = torch.cat([h(f) for h, f in zip(self.class_head, (f1, f2, f3))], dim=1)
        # Concatenate the outputs from the OBB and angle heads
        obbs = torch.cat([h(f) for h, f in zip(self.obb_head, (f1, f2, f3))], dim=1)
        # Concatenate the outputs from the angle heads
        angs = torch.cat([h(f) for h, f in zip(self.angle_head, (f1, f2, f3))], dim=1)
        return logits, obbs, angs


class ViTFeature2D(nn.Module):
    """
    Wrapper around a Vision Transformer (ViT) feature extractor that converts sequence outputs
    (flattened tokens) into spatial 2D feature maps, excluding the [CLS] token.
    """

    def __init__(self, seq_extractor: nn.Module, patch_size: int):
        """
        Initializes the ViTFeature2D module.

        Args:
            seq_extractor (nn.Module): ViT-based sequence feature extractor.
            patch_size (int): Size of the patch used in ViT (e.g., 16).
        """
        super().__init__()
        self.seq_extractor = seq_extractor
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass to convert token sequences into 2D feature maps.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping feature names to 2D feature maps.
        """
        out = self.seq_extractor(x)
        maps = {}
        for name, seq in out.items():
            seq = seq[:, 1:, :]  # Remove [CLS] token
            B, L, C = seq.shape
            H = W = int(L**0.5)
            feat2d = seq.permute(0, 2, 1).reshape(B, C, H, W)
            maps[name] = feat2d
        return maps
