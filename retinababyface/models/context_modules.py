from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp: int, oup: int, stride: int = 1, leaky: float = 0) -> nn.Sequential:
    """
    Creates a convolutional layer followed by batch normalization and LeakyReLU activation.

    Args:
        inp (int): Number of input channels.
        oup (int): Number of output channels.
        stride (int): Stride of the convolution. Defaults to 1.
        leaky (float): Negative slope for LeakyReLU. Defaults to 0.

    Returns:
        nn.Sequential: A sequential container of the convolutional, batch normalization, and LeakyReLU layers.
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),  # Convolutional layer.
        nn.BatchNorm2d(oup),  # Batch normalization layer.
        nn.LeakyReLU(negative_slope=leaky, inplace=True),  # LeakyReLU activation.
    )


def conv_bn_no_relu(inp: int, oup: int, stride: int) -> nn.Sequential:
    """
    Creates a convolutional layer followed by batch normalization, without ReLU activation.

    Args:
        inp (int): Number of input channels.
        oup (int): Number of output channels.
        stride (int): Stride of the convolution.

    Returns:
        nn.Sequential: A sequential container of the convolutional and batch normalization layers.
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),  # Convolutional layer.
        nn.BatchNorm2d(oup),  # Batch normalization layer.
    )


def conv_bn1X1(inp: int, oup: int, stride: int, leaky: float = 0) -> nn.Sequential:
    """
    Creates a 1x1 convolutional layer followed by batch normalization and LeakyReLU activation.

    Args:
        inp (int): Number of input channels.
        oup (int): Number of output channels.
        stride (int): Stride of the convolution.
        leaky (float): Negative slope for LeakyReLU. Defaults to 0.

    Returns:
        nn.Sequential: A sequential container of the 1x1 convolutional, batch normalization, and LeakyReLU layers.
    """
    return nn.Sequential(
        nn.Conv2d(
            inp, oup, 1, stride, padding=0, bias=False
        ),  # 1x1 convolutional layer.
        nn.BatchNorm2d(oup),  # Batch normalization layer.
        nn.LeakyReLU(negative_slope=leaky, inplace=True),  # LeakyReLU activation.
    )


def conv_dw(inp: int, oup: int, stride: int, leaky: float = 0.1) -> nn.Sequential:
    """
    Creates a depthwise separable convolutional layer followed by batch normalization and LeakyReLU activation.

    Args:
        inp (int): Number of input channels.
        oup (int): Number of output channels.
        stride (int): Stride of the convolution.
        leaky (float): Negative slope for LeakyReLU. Defaults to 0.1.

    Returns:
        nn.Sequential: A sequential container of the depthwise and pointwise convolutional, batch normalization, and LeakyReLU layers.
    """
    return nn.Sequential(
        nn.Conv2d(
            inp, inp, 3, stride, 1, groups=inp, bias=False
        ),  # Depthwise convolutional layer.
        nn.BatchNorm2d(inp),  # Batch normalization layer.
        nn.LeakyReLU(negative_slope=leaky, inplace=True),  # LeakyReLU activation.
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),  # Pointwise convolutional layer.
        nn.BatchNorm2d(oup),  # Batch normalization layer.
        nn.LeakyReLU(negative_slope=leaky, inplace=True),  # LeakyReLU activation.
    )


class SSH(nn.Module):
    """
    SSH (Single Stage Headless) module, used for feature aggregation.
    """

    def __init__(self, in_channel: int, out_channel: int):
        """
        Initializes the SSH module.

        Args:
            in_channel (int): Number of input channels.
            out_channel (int): Number of output channels.
        """
        super(SSH, self).__init__()
        assert out_channel % 4 == 0  # Ensure output channels are divisible by 4.
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(
            in_channel, out_channel // 2, stride=1
        )  # 3x3 convolution.

        self.conv5X5_1 = conv_bn(
            in_channel, out_channel // 4, stride=1, leaky=leaky
        )  # 5x5 convolution (part 1).
        self.conv5X5_2 = conv_bn_no_relu(
            out_channel // 4, out_channel // 4, stride=1
        )  # 5x5 convolution (part 2).

        self.conv7X7_2 = conv_bn(
            out_channel // 4, out_channel // 4, stride=1, leaky=leaky
        )  # 7x7 convolution (part 2).
        self.conv7x7_3 = conv_bn_no_relu(
            out_channel // 4, out_channel // 4, stride=1
        )  # 7x7 convolution (part 3).

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SSH module.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        conv5X5_1 = self.conv5X5_1(input)
        return F.relu(
            torch.cat(
                [
                    self.conv3X3(input),
                    self.conv5X5_2(conv5X5_1),
                    self.conv7x7_3(self.conv7X7_2(conv5X5_1)),
                ],
                dim=1,
            )
        )


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) module for multi-scale feature fusion.

    This module takes feature maps from different levels of a backbone network
    and performs top-down and lateral connections to create a feature pyramid.
    It outputs multi-scale feature maps for further processing.

    Args:
        in_channels_list (List[int]): List of input channel numbers for each feature level.
        out_channels (int): Number of output channels for the feature pyramid.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int):
        """
        Initializes the FPN module.

        Args:
            in_channels_list (List[int]): List of input channel numbers for each feature level.
            out_channels (int): Number of output channels for the feature pyramid.
        """
        super(FPN, self).__init__()
        if len(in_channels_list) != 4:
            raise ValueError(
                f"Expected 'in_channels_list' to have exactly 4 elements, but got {len(in_channels_list)}."
            )

        # Determine the negative slope for LeakyReLU based on the number of output channels.
        leaky = 0.1 if out_channels <= 64 else 0.0

        # Lateral 1x1 convolutions for each feature level.
        self.lateral2 = conv_bn1X1(
            in_channels_list[0], out_channels, stride=1, leaky=leaky
        )
        self.lateral3 = conv_bn1X1(
            in_channels_list[1], out_channels, stride=1, leaky=leaky
        )
        self.lateral4 = conv_bn1X1(
            in_channels_list[2], out_channels, stride=1, leaky=leaky
        )
        self.lateral5 = conv_bn1X1(
            in_channels_list[3], out_channels, stride=1, leaky=leaky
        )

        # Smooth 3x3 convolutions for each feature level.
        self.smooth2 = conv_bn(out_channels, out_channels, stride=1, leaky=leaky)
        self.smooth3 = conv_bn(out_channels, out_channels, stride=1, leaky=leaky)
        self.smooth4 = conv_bn(out_channels, out_channels, stride=1, leaky=leaky)
        self.smooth5 = conv_bn(out_channels, out_channels, stride=1, leaky=leaky)

        # Downsampling convolution for generating P6 from P5.
        self.p6_conv = conv_bn(out_channels, out_channels, stride=2, leaky=leaky)

    def forward(self, c_feats: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass of the FPN module.

        Args:
            c_feats (Dict[str, torch.Tensor]): Dictionary containing feature maps from the backbone network.
                Keys are ["feat1", "feat2", "feat3", "feat4"], corresponding to C2, C3, C4, and C5 respectively.

        Returns:
            List[torch.Tensor]: List of feature maps [P2, P3, P4, P5, P6] after multi-scale fusion.
        """
        # Unpack the feature maps from the input dictionary.
        c2 = c_feats["feat1"]  # Feature map from C2 (lowest level).
        c3 = c_feats["feat2"]  # Feature map from C3.
        c4 = c_feats["feat3"]  # Feature map from C4.
        c5 = c_feats["feat4"]  # Feature map from C5 (highest level).

        # Step 1: Apply lateral 1x1 convolutions to each feature map.
        p5_lat = self.lateral5(c5)  # Lateral convolution for C5.
        p4_lat = self.lateral4(c4)  # Lateral convolution for C4.
        p3_lat = self.lateral3(c3)  # Lateral convolution for C3.
        p2_lat = self.lateral2(c2)  # Lateral convolution for C2.

        # Step 2: Perform top-down fusion and apply smooth 3x3 convolutions.
        # P5 final = smooth5(p5_lat)
        p5 = self.smooth5(p5_lat)

        # P4 = smooth4(p4_lat + upsample(p5_lat))
        p5_upsampled_to_p4 = F.interpolate(
            p5_lat, size=p4_lat.shape[-2:], mode="nearest"
        )  # Upsample P5 to match P4's size.
        p4 = self.smooth4(
            p4_lat + p5_upsampled_to_p4
        )  # Fuse P4 and upsampled P5, then smooth.

        # P3 = smooth3(p3_lat + upsample(p4_lat + p5_upsampled_to_p4))
        fused_4 = p4_lat + p5_upsampled_to_p4  # Fuse P4 and upsampled P5.
        p4_upsampled_to_p3 = F.interpolate(
            fused_4, size=p3_lat.shape[-2:], mode="nearest"
        )  # Upsample fused P4 to match P3's size.
        p3 = self.smooth3(
            p3_lat + p4_upsampled_to_p3
        )  # Fuse P3 and upsampled fused P4, then smooth.

        # P2 = smooth2(p2_lat + upsample(p3_lat + p4_upsampled_to_p3))
        fused_3 = p3_lat + p4_upsampled_to_p3  # Fuse P3 and upsampled fused P4.
        p3_upsampled_to_p2 = F.interpolate(
            fused_3, size=p2_lat.shape[-2:], mode="nearest"
        )  # Upsample fused P3 to match P2's size.
        p2 = self.smooth2(
            p2_lat + p3_upsampled_to_p2
        )  # Fuse P2 and upsampled fused P3, then smooth.

        # Step 3: Generate P6 by downsampling P5.
        p6 = self.p6_conv(p5)  # Downsample P5 to create P6.

        # Return the list of feature maps [P2, P3, P4, P5, P6].
        return [p2, p3, p4, p5, p6]


class MobileNetV1(nn.Module):
    """
    MobileNetV1 backbone network.
    """

    def __init__(self):
        """
        Initializes the MobileNetV1 module.
        """
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky=0.1),  # Stage 1.
            conv_dw(8, 16, 1),
            conv_dw(16, 32, 2),
            conv_dw(32, 32, 1),
            conv_dw(32, 64, 2),
            conv_dw(64, 64, 1),
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # Stage 2.
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),  # Stage 3.
            conv_dw(256, 256, 1),
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive average pooling.
        self.fc = nn.Linear(256, 1000)  # Fully connected layer.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MobileNetV1 module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.fc(self.avg(self.stage3(self.stage2(self.stage1(x)))).view(-1, 256))
