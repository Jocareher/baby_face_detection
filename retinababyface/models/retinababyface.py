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

from .context_modules import FPN, SSH, MobileNetV1
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
        return self.conv(x).permute(0, 2, 3, 1).contiguous().view(
            x.size(0), -1, 8
        ) # torch.tanh(
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
    RetinaBabyFace model integrating backbone, FPN, SSH blocks,
    and multiple prediction heads for oriented bounding box detection, angle estimation, and class prediction.

    This model is designed for face detection and classification tasks, with additional capabilities
    for predicting oriented bounding boxes (OBBs) and rotation angles.
    """

    def __init__(
        self,
        backbone_name: str = "densenet121",
        out_channel: int = 64,
        pretrained: bool = True,
    ):
        """
        Initializes the RetinaBabyFace model.

        Args:
            backbone_name (str): Name of the backbone to use (e.g., "mobilenetv1", "resnet50", "vgg16", "densenet121", "vit").
            out_channel (int): Number of output channels for FPN layers.
            pretrained (bool): Whether to load pretrained weights for the backbone.
        """
        super().__init__()

        # Build backbone and retrieve feature extractor, return layers, and in_channels_list
        self.backbone, return_layers, in_channels_list = self.make_backbone(
            backbone_name, pretrained
        )

        # Feature Pyramid Network (FPN) for multi-scale feature aggregation
        self.fpn = FPN(in_channels_list, out_channel)

        # SSH layers applied on each FPN output for enhanced feature extraction
        self.ssh1_stage1 = SSH(out_channel, out_channel)  # P3
        self.ssh1_stage2 = SSH(out_channel, out_channel)

        self.ssh2_stage1 = SSH(out_channel, out_channel)  # P4
        self.ssh2_stage2 = SSH(out_channel, out_channel)

        self.ssh3_stage1 = SSH(out_channel, out_channel)  # P5
        self.ssh3_stage2 = SSH(out_channel, out_channel)

        self.ssh4_stage1 = SSH(out_channel, out_channel)  # P2
        self.ssh4_stage2 = SSH(out_channel, out_channel)

        self.ssh5_stage1 = SSH(out_channel, out_channel)  # P6
        self.ssh5_stage2 = SSH(out_channel, out_channel)

        # Prediction heads:
        # - OBB head for predicting 8 vertex displacements of oriented bounding boxes
        # - Angle head for predicting rotation angles of bounding boxes
        # - Class head for predicting class logits
        # - Face head for predicting face/no-face logits
        self.obb_head = OBBHead(out_channel)
        self.angle_head = AngleHead(out_channel)
        self.class_head = ClassHead(out_channel)
        self.face_head = FaceHead(out_channel)

    def make_backbone(
        self, name: str, pretrained: bool
    ) -> tuple[nn.Module, dict, list[int]]:
        """
        Creates and returns a feature extractor from a specified backbone.

        Args:
            name (str): Name of the backbone model (e.g., "resnet50", "vgg16", "densenet121", "vit", "mobilenetv1").
            pretrained (bool): Whether to use pretrained weights.

        Returns:
            Tuple containing:
                - feature extractor (nn.Module): Backbone feature extractor.
                - return_layers (dict): Mapping of layer names to output names.
                - in_channels_list (list[int]): Channels for each returned feature map.
        """
        if name == "resnet50":
            # ResNet50 backbone
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights)
            return_layers = {
                "layer1": "feat1",  # C2
                "layer2": "feat2",  # C3
                "layer3": "feat3",  # C4
                "layer4": "feat4",  # C5
            }
            in_channels_list = [256, 512, 1024, 2048]
            feat_ext = create_feature_extractor(model, return_layers)

        elif name == "vgg16":
            # VGG16 backbone
            weights = VGG16_Weights.DEFAULT if pretrained else None
            model = vgg16(weights=weights).features
            return_layers = {
                "4": "feat1",  # C2
                "9": "feat2",  # C3
                "16": "feat3",  # C4
                "23": "feat4",  # C5
            }
            in_channels_list = [128, 256, 512, 512]
            feat_ext = create_feature_extractor(model, return_layers)

        elif name == "densenet121":
            # DenseNet121 backbone
            weights = DenseNet121_Weights.DEFAULT if pretrained else None
            model = densenet121(weights=weights).features
            return_layers = {
                "denseblock1": "feat1",  # C2
                "denseblock2": "feat2",  # C3
                "denseblock3": "feat3",  # C4
                "denseblock4": "feat4",  # C5
            }
            in_channels_list = [256, 512, 1024, 1024]
            feat_ext = create_feature_extractor(model, return_layers)

        elif name == "vit":
            # Vision Transformer (ViT) backbone
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            vit = vit_b_16(weights=weights)
            return_layers = {
                "encoder.layers.encoder_layer_2": "feat1",  # C2
                "encoder.layers.encoder_layer_5": "feat2",  # C3
                "encoder.layers.encoder_layer_8": "feat3",  # C4
                "encoder.layers.encoder_layer_11": "feat4",  # C5
            }
            in_channels_list = [768, 768, 768, 768]
            feat_seq = create_feature_extractor(vit, return_layers)
            feat_ext = ViTFeature2D(feat_seq, patch_size=16)

        else:
            # MobileNetV1 backbone
            model = MobileNetV1()
            return_layers = {
                "stage1": "feat1",  # C2: channels=32
                "stage2": "feat2",  # C3: channels=64
                "stage3": "feat3",  # C4: channels=128
                "stage4": "feat4",  # C5: channels=256
            }
            in_channels_list = [32, 64, 128, 256]
            feat_ext = create_feature_extractor(model, return_layers)

        return feat_ext, return_layers, in_channels_list

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the RetinaBabyFace model.

        This method performs multi-scale feature extraction, context refinement, and prediction
        across multiple levels of the feature pyramid. It integrates outputs from two stages
        of SSH refinement and combines predictions from both stages for final outputs.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - orientation_logits (torch.Tensor): Class logits for orientation prediction.
                - face_logits (torch.Tensor): Binary logits for face/no-face prediction.
                - obbs (torch.Tensor): Oriented bounding box (OBB) vertex displacements.
                - angs (torch.Tensor): Rotation angles of OBBs wrapped to [-π, π].
        """
        # 1) Extract 4 feature maps C2..C5 from the backbone
        feats = self.backbone(x)

        # 2) Pass through FPN → obtain P2..P6 feature pyramid levels
        p2, p3, p4, p5, p6 = self.fpn(feats)

        # 3) Context stage 1 (SSH stage1) applied to each pyramid level
        c2_1 = self.ssh4_stage1(p2)  # P2 → c2_1
        c3_1 = self.ssh1_stage1(p3)  # P3 → c3_1
        c4_1 = self.ssh2_stage1(p4)  # P4 → c4_1
        c5_1 = self.ssh3_stage1(p5)  # P5 → c5_1
        c6_1 = self.ssh5_stage1(p6)  # P6 → c6_1

        # 4) Intermediate predictions from stage 1
        cls2_1 = self.class_head(c2_1)  # Class logits for P2
        face2_1 = self.face_head(c2_1)  # Face logits for P2
        obb2_1 = self.obb_head(c2_1)  # OBB vertex displacements for P2
        ang2_1 = self.angle_head(c2_1)  # Rotation angles for P2

        cls3_1 = self.class_head(c3_1)  # Class logits for P3
        face3_1 = self.face_head(c3_1)  # Face logits for P3
        obb3_1 = self.obb_head(c3_1)  # OBB vertex displacements for P3
        ang3_1 = self.angle_head(c3_1)  # Rotation angles for P3

        cls4_1 = self.class_head(c4_1)  # Class logits for P4
        face4_1 = self.face_head(c4_1)  # Face logits for P4
        obb4_1 = self.obb_head(c4_1)  # OBB vertex displacements for P4
        ang4_1 = self.angle_head(c4_1)  # Rotation angles for P4

        cls5_1 = self.class_head(c5_1)  # Class logits for P5
        face5_1 = self.face_head(c5_1)  # Face logits for P5
        obb5_1 = self.obb_head(c5_1)  # OBB vertex displacements for P5
        ang5_1 = self.angle_head(c5_1)  # Rotation angles for P5

        cls6_1 = self.class_head(c6_1)  # Class logits for P6
        face6_1 = self.face_head(c6_1)  # Face logits for P6
        obb6_1 = self.obb_head(c6_1)  # OBB vertex displacements for P6
        ang6_1 = self.angle_head(c6_1)  # Rotation angles for P6

        # 5) Context stage 2 (SSH stage2) → refinement
        c2_2 = self.ssh4_stage2(c2_1)  # Refined P2
        c3_2 = self.ssh1_stage2(c3_1)  # Refined P3
        c4_2 = self.ssh2_stage2(c4_1)  # Refined P4
        c5_2 = self.ssh3_stage2(c5_1)  # Refined P5
        c6_2 = self.ssh5_stage2(c6_1)  # Refined P6

        # 6) Predictions from stage 2
        cls2_2 = self.class_head(c2_2)  # Refined class logits for P2
        face2_2 = self.face_head(c2_2)  # Refined face logits for P2
        obb2_2 = self.obb_head(c2_2)  # Refined OBB vertex displacements for P2
        ang2_2 = self.angle_head(c2_2)  # Refined rotation angles for P2

        cls3_2 = self.class_head(c3_2)  # Refined class logits for P3
        face3_2 = self.face_head(c3_2)  # Refined face logits for P3
        obb3_2 = self.obb_head(c3_2)  # Refined OBB vertex displacements for P3
        ang3_2 = self.angle_head(c3_2)  # Refined rotation angles for P3

        cls4_2 = self.class_head(c4_2)  # Refined class logits for P4
        face4_2 = self.face_head(c4_2)  # Refined face logits for P4
        obb4_2 = self.obb_head(c4_2)  # Refined OBB vertex displacements for P4
        ang4_2 = self.angle_head(c4_2)  # Refined rotation angles for P4

        cls5_2 = self.class_head(c5_2)  # Refined class logits for P5
        face5_2 = self.face_head(c5_2)  # Refined face logits for P5
        obb5_2 = self.obb_head(c5_2)  # Refined OBB vertex displacements for P5
        ang5_2 = self.angle_head(c5_2)  # Refined rotation angles for P5

        cls6_2 = self.class_head(c6_2)  # Refined class logits for P6
        face6_2 = self.face_head(c6_2)  # Refined face logits for P6
        obb6_2 = self.obb_head(c6_2)  # Refined OBB vertex displacements for P6
        ang6_2 = self.angle_head(c6_2)  # Refined rotation angles for P6

        # 7) Combine stage1 + stage2 predictions (additive refinement)
        cls2_f = cls2_1 + cls2_2
        face2_f = face2_1 + face2_2
        obb2_f = obb2_1 + obb2_2
        ang2_f = wrap_to_pi(ang2_1 + ang2_2)

        cls3_f = cls3_1 + cls3_2
        face3_f = face3_1 + face3_2
        obb3_f = obb3_1 + obb3_2
        ang3_f = wrap_to_pi(ang3_1 + ang3_2)

        cls4_f = cls4_1 + cls4_2
        face4_f = face4_1 + face4_2
        obb4_f = obb4_1 + obb4_2
        ang4_f = wrap_to_pi(ang4_1 + ang4_2)

        cls5_f = cls5_1 + cls5_2
        face5_f = face5_1 + face5_2
        obb5_f = obb5_1 + obb5_2
        ang5_f = wrap_to_pi(ang5_1 + ang5_2)

        cls6_f = cls6_1 + cls6_2
        face6_f = face6_1 + face6_2
        obb6_f = obb6_1 + obb6_2
        ang6_f = wrap_to_pi(ang6_1 + ang6_2)

        # 8) Concatenate predictions across the 5 pyramid levels
        orientation_logits = torch.cat([cls2_f, cls3_f, cls4_f, cls5_f, cls6_f], dim=1)
        face_logits = torch.cat([face2_f, face3_f, face4_f, face5_f, face6_f], dim=1)
        obbs = torch.cat([obb2_f, obb3_f, obb4_f, obb5_f, obb6_f], dim=1)
        angs = torch.cat([ang2_f, ang3_f, ang4_f, ang5_f, ang6_f], dim=1)

        return orientation_logits, face_logits, obbs, angs


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
        model (nn.Module): The model containing the submodules `obb_head`, `angle_head`, and `class_head`.

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
    freeze: bool,
    fine_tuning: bool,
    last_block_tokens=("layer4", "denseblock4", "stage4", "encoder.layers.encoder_layer_11"),
):
    """
    Adjusts the training state of the backbone parameters in the model.

    This function allows freezing or unfreezing the backbone parameters, with an optional fine-tuning mode
    that keeps the last block trainable while freezing the rest. Additionally, it handles BatchNorm layers
    by setting them to evaluation mode and freezing their affine parameters when the backbone is frozen.

    Args:
        model (nn.Module): The model containing the backbone.
        freeze (bool): Whether to freeze the backbone parameters.
        fine_tuning (bool): Whether to enable fine-tuning mode (keeps the last block trainable).
        last_block_tokens (tuple[str]): Tokens identifying the last block layers in the backbone.

    Returns:
        None
    """
    # 1) Freeze or unfreeze **all** backbone parameters
    for name, p in model.backbone.named_parameters():
        p.requires_grad = not freeze

    # 2) If fine-tuning is enabled, re-activate the parameters of the last block
    if freeze and fine_tuning:
        for name, p in model.backbone.named_parameters():
            if any(tok in name for tok in last_block_tokens):
                p.requires_grad = True

    # 3) If freeze is True, set all BatchNorm layers to evaluation mode and freeze their affine parameters
    if freeze:
        for module in model.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()
                module.weight.requires_grad = False
                module.bias.requires_grad = False

        # 4) If fine-tuning is enabled, re-activate BatchNorm layers in the last block
        if fine_tuning:
            for name, module in model.backbone.named_modules():
                if isinstance(module, nn.BatchNorm2d) and any(tok in name for tok in last_block_tokens):
                    module.train()
                    module.weight.requires_grad = True
                    module.bias.requires_grad = True
