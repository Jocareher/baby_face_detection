from typing import Optional, Dict, Tuple
import math
import os

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

from .context_modules import FPN, SSH, MobileNetV1
from .vggface_backbone import vggface2_resnet50
from .arcface_backbone import arcface_backbone
from data_setup.augmentations import wrap_to_pi
import config


class FaceHead(nn.Module):
    """
    Head module for predicting a binary face/no‐face logit per anchor.
    The output is a single value per anchor, indicating the presence of a face.


    Output shape:
        - Input: (B, C, H, W)
        - Output: (B, N, 1) where N = H × W × num_anchors
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, config.NUM_ANCHORS * 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prob = (
            self.conv(x).permute(0, 2, 3, 1).contiguous()
        )  # (B, H * W * num_anchors, 1)
        return prob.view(x.size(0), -1, 1)


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

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, config.NUM_ANCHORS * 8, kernel_size=1)

    def forward(self, x):
        # Apply 1x1 convolution, reshape and apply tanh to constrain output to [-1, 1]
        # The output shape is (B, num_anchors * H * W, 8)
        # The 8 values correspond to the 4 vertices of the OBB.
        # The vertices are represented as (Δx1, Δy1, Δx2, Δy2, Δx3, Δy3, Δx4, Δy4)
        # The output is reshaped to (B, N, 8) where N = H × W × num_anchors
        return (
            self.conv(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 8)
        )  # torch.tanh(
        # self.conv(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 8)


class AngleHead(nn.Module):
    """
    Head module for predicting the rotation angle of the OBB.
    The output is constrained to the range [0, 2π] using a sigmoid activation,
    so that the angle remains within a reasonable distance from the anchor.
    Output shape:
        - Input: (B, C, H, W)
        - Output: (B, N, 1) where N = H × W × num_anchors
    The angle is represented in radians.
    The output is then wrapped to the range [-π, π] using wrap_to_pi function.
    The output is reshaped to (B, N, 1) where N = H × W × num_anchors
    The 1 value corresponds to the rotation angle of the OBB.

    """

    def __init__(self, inchannels: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(inchannels, config.NUM_ANCHORS, kernel_size=1)

    def forward(self, x):
        # Apply 1x1 convolution, reshape and apply sigmoid to constrain output to [0, 2π]
        # The output shape is (B, num_anchors * H * W, 1)
        # The 1 value corresponds to the rotation angle of the OBB.
        # The output is reshaped to (B, N, 1) where N = H × W × num_anchors
        # The angle is represented in radians.
        # The output is then wrapped to the range [-π, π] using wrap_to_pi function.
        # The output is reshaped to (B, N, 1) where N = H × W × num_anchors
        raw = torch.sigmoid(self.conv(x).permute(0, 2, 3, 1).contiguous()) * 2 * math.pi
        return wrap_to_pi(raw).view(x.size(0), -1, 1)


class ClassHead(nn.Module):
    """
    Head module for class prediction.
    """

    def __init__(self, inchannels: int = 64, num_classes: int = 5):
        """
        Initializes the ClassHead module.

        Args:
            inchannels (int): Number of input channels. Defaults to 64.
            num_classes (int): Number of classes to predict. Defaults to 5.
            num_anchors (int): Number of anchors per location. Defaults to 2.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            inchannels, config.NUM_ANCHORS * num_classes, kernel_size=1
        )  # 1x1 convolution for class prediction.
        self.num_classes = num_classes  # Store the number of classes.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ClassHead module.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Predicted class orientation_logits.
        """
        # Apply the convolution and rearrange the tensor dimensions.
        # The output shape is (batch_size, num_anchors * H * W, num_classes).
        # The num_classes values correspond to the class orientation_logits for each anchor.
        # The orientation_logits are not normalized, so they can be used directly for classification.
        return (
            self.conv(x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(x.size(0), -1, self.num_classes)
        )


class RetinaBabyFace(nn.Module):
    """
    RetinaBabyFace model for face detection, orientation estimation and pose classification.

    This model combines:
    - A feature extraction backbone (various options like ResNet50, DenseNet121, ViT, etc.)
    - Feature Pyramid Network (FPN) for multi-scale features
    - SSH context modules for feature refinement
    - Multiple prediction heads:
        - Face head for binary face/no-face detection
        - OBB head for oriented bounding box regression
        - Angle head for rotation angle estimation
        - Class head for pose classification

    The model processes features at 5 scales (P2-P6) through two SSH stages for refinement.
    Predictions from both stages are combined additively for the final output.

    Output Shapes:
        - orientation_logits: (batch_size, num_anchors_total, num_classes)
        - face_logits: (batch_size, num_anchors_total, 1)
        - obbs: (batch_size, num_anchors_total, 8) - 4 pairs of (x,y) vertex offsets
        - angs: (batch_size, num_anchors_total, 1) - rotation angles in [-π, π]

    Supported backbones:
        - ResNet50 (ImageNet or VGGFace2 pretrained)
        - DenseNet121
        - VGG16
        - Vision Transformer (ViT)
        - MobileNetV1
        - ArcFace IR-SE-50
    """


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


def reset_heads(model: nn.Module) -> None:
    """
    Reinitializes the weights and biases of the model's prediction heads.

    This function applies Kaiming-He (He normal) initialization to the convolutional layers
    of the OBB regression head (`obb_head`), angle regression head (`angle_head`), and
    classification head (`class_head`). Biases are reset to zero.

    This is typically used to reinitialize the heads before fine-tuning or after structural changes
    to the model.

    Args:
        model (nn.Module): The model containing the submodules `obb_head`, `angle_head`, `class_head` and `face_head`.

    Returns:
        None
    """
    for head in [model.obb_head, model.angle_head, model.class_head, model.face_head]:
        for layer in head.modules():
            if isinstance(layer, nn.Conv2d):
                # Apply Kaiming-He initialization for convolution weights
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="relu"
                )
                # Initialize bias to zero if present
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


def set_backbone_frozen(
    model: nn.Module,
    mode: str = "feature_extract",  # "feature_extract" | "fine_tune" | "train_all"
    last_block_tokens=(
        "denseblock4",
        "layer4",
        "stage4",
        "encoder.layers.encoder_layer_11",
    ),
):
    """
    Adjusts which parts of the backbone are trainable.

    This function allows for flexible control over the training behavior of the backbone
    in a neural network model. It supports three modes:
      - "feature_extract": All backbone layers are frozen, and BatchNorm layers are set to evaluation mode.
      - "fine_tune": Only the last block of the backbone and its BatchNorm layers are trainable.
      - "train_all": All backbone layers are trainable, including BatchNorm layers.

    Args:
        model (nn.Module): The model containing the backbone.
        mode (str): Training mode for the backbone. Options are:
            - "feature_extract": Freeze all layers.
            - "fine_tune": Train only the last block.
            - "train_all": Train all layers.
        last_block_tokens (tuple): Identifiers for the last block layers in the backbone.
            These tokens are used to determine which layers belong to the last block.

    Returns:
        None
    """
    assert mode in {
        "feature_extractor",
        "fine_tuning",
        "train_all",
    }, "Invalid mode. Choose from 'feature_extractor', 'fine_tuning', or 'train_all'."

    freeze_all = mode in {"feature_extractor", "fine_tuning"}

    # 1) Freeze or unfreeze gradients for backbone parameters
    for name, p in model.backbone.named_parameters():
        p.requires_grad = not freeze_all

    # 2) In "fine_tune" mode, re-enable gradients for the last block
    if mode == "fine_tuning":
        for name, p in model.backbone.named_parameters():
            if any(tok in name for tok in last_block_tokens):
                p.requires_grad = True  # Enable gradients for convolutional and BatchNorm layers in the last block

    # 3) Handle BatchNorm layers based on the mode
    for name, m in model.backbone.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if mode == "feature_extract":
                m.eval()  # Set BatchNorm layers to evaluation mode (fixed statistics)
                m.weight.requires_grad = False  # Freeze BatchNorm weights
                m.bias.requires_grad = False  # Freeze BatchNorm biases
            elif mode == "fine_tuning":
                if any(tok in name for tok in last_block_tokens):
                    m.train()  # BatchNorm layers in the last block update statistics
                else:
                    m.eval()  # Other BatchNorm layers remain fixed
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
            else:  # "train_all" mode
                m.train()  # All BatchNorm layers update statistics and are trainable
