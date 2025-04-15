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
        conv3X3 = self.conv3X3(input)  # 3x3 convolution.

        conv5X5_1 = self.conv5X5_1(input)  # 5x5 convolution (part 1).
        conv5X5 = self.conv5X5_2(conv5X5_1)  # 5x5 convolution (part 2).

        conv7X7_2 = self.conv7X7_2(conv5X5_1)  # 7x7 convolution (part 2).
        conv7X7 = self.conv7x7_3(conv7X7_2)  # 7x7 convolution (part 3).

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)  # Concatenate the outputs.
        out = F.relu(out)  # Apply ReLU activation.
        return out


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) module for multi-scale feature fusion.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int):
        """
        Initializes the FPN module.

        Args:
            in_channels_list (List[int]): List of input channel numbers for each feature level.
            out_channels (int): Number of output channels.
        """
        super(FPN, self).__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = conv_bn1X1(
            in_channels_list[0], out_channels, stride=1, leaky=leaky
        )  # 1x1 convolution for feature level 1.
        self.output2 = conv_bn1X1(
            in_channels_list[1], out_channels, stride=1, leaky=leaky
        )  # 1x1 convolution for feature level 2.
        self.output3 = conv_bn1X1(
            in_channels_list[2], out_channels, stride=1, leaky=leaky
        )  # 1x1 convolution for feature level 3.

        self.merge1 = conv_bn(
            out_channels, out_channels, leaky=leaky
        )  # Merge convolution for feature level 1.
        self.merge2 = conv_bn(
            out_channels, out_channels, leaky=leaky
        )  # Merge convolution for feature level 2.

    def forward(self, input: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass of the FPN module.

        Args:
            input (Dict[str, torch.Tensor]): Dictionary of input feature maps.

        Returns:
            List[torch.Tensor]: List of output feature maps.
        """
        input = list(input.values())  # Get the values of the input dictionary.

        output1 = self.output1(input[0])  # Process feature level 1.
        output2 = self.output2(input[1])  # Process feature level 2.
        output3 = self.output3(input[2])  # Process feature level 3.

        up3 = F.interpolate(
            output3, size=[output2.size(2), output2.size(3)], mode="nearest"
        )  # Upsample feature level 3.
        output2 = output2 + up3  # Add upsampled feature level 3 to feature level 2.
        output2 = self.merge2(output2)  # Merge feature level 2.

        up2 = F.interpolate(
            output2, size=[output1.size(2), output1.size(3)], mode="nearest"
        )  # Upsample feature level 2.
        output1 = output1 + up2  # Add upsampled feature level 2 to feature level 1.
        output1 = self.merge1(output1)  # Merge feature level 1.

        out = [output1, output2, output3]  # Create the output list.
        return out


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
        x = self.stage1(x)  # Stage 1.
        x = self.stage2(x)  # Stage 2.
        x = self.stage3(x)  # Stage 3.
        x = self.avg(x)  # Adaptive average pooling.
        x = x.view(-1, 256)  # Reshape the tensor.
        x = self.fc(x)  # Fully connected layer.
        return x
