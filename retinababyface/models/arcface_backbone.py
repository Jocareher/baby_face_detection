from typing import Optional
import re

from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
from torch import nn
import torch
from collections import namedtuple

##################################  Original Arcface Model #############################################################
# Taken and adapted from https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/model.py


# ---------- Utility Classes and Functions ----------
class Flatten(nn.Module):
    """Flattens input tensor to 2D."""

    def forward(self, x):
        return x.view(x.size(0), -1)


def l2_norm(x, axis=1):
    """
    Performs L2 normalization along specified axis.
    Args:
        x: Input tensor
        axis: Dimension to normalize over
    Returns:
        Normalized tensor
    """
    return torch.div(x, torch.norm(x, 2, axis, keepdim=True))


# ---------- Squeeze-and-Excitation and Bottleneck Blocks ----------
class SEModule(nn.Module):
    """
    Squeeze-and-Excitation module for channel attention.
    Args:
        c: Number of input channels
        r: Reduction ratio for bottleneck
    """

    def __init__(self, c, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.avg(x))


class bottleneck_IR(nn.Module):
    """
    Basic IR bottleneck block.
    Args:
        in_c: Input channels
        out_c: Output channels
        s: Stride
    """

    def __init__(self, in_c, out_c, s):
        super().__init__()
        if in_c == out_c:
            self.short = nn.MaxPool2d(1, s)
        else:
            self.short = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, s, bias=False), nn.BatchNorm2d(out_c)
            )

        self.res = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.PReLU(out_c),
            nn.Conv2d(out_c, out_c, 3, s, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        return self.res(x) + self.short(x)


class bottleneck_IR_SE(bottleneck_IR):
    """IR bottleneck block with Squeeze-and-Excitation."""

    def __init__(self, in_c, out_c, s):
        super().__init__(in_c, out_c, s)
        self.res.add_module("se", SEModule(out_c))


# ---------- IR-SE-50 Backbone Architecture ----------
Blk = namedtuple("Blk", ["in_c", "out_c", "s"])


def _blocks():
    """Defines the block configuration for IR-SE-50."""
    cfg = [
        (64, 64, 3, 1),  # (in_channels, out_channels, num_blocks, stride)
        (64, 128, 4, 2),
        (128, 256, 14, 2),
        (256, 512, 3, 2),
    ]
    blocks = []
    for in_c, out_c, n, s in cfg:
        blocks.append([Blk(in_c, out_c, s)] + [Blk(out_c, out_c, 1)] * (n - 1))
    return blocks


class BackboneIRSE50(nn.Module):
    """
    IR-SE-50 backbone network for face recognition.
    Features:
    - Uses IR blocks with Squeeze-and-Excitation
    - Outputs feature maps with channels [64, 128, 256, 512]
    """

    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.MaxPool2d(2),
        )

        modules = []
        for block_list in _blocks():
            for block in block_list:
                modules.append(bottleneck_IR_SE(block.in_c, block.out_c, block.s))
        self.body = nn.Sequential(*modules)

        # Channel dimensions for feature pyramid network (FPN)
        self.out_channels = [64, 128, 256, 512]

    def forward(self, x):  # x shape: (batch_size, 3, height, width)
        x = self.input(x)
        return self.body(x)  # Returns final convolutional feature map


def arcface_backbone(weights: Optional[str] = None) -> nn.Module:
    """
    Creates and optionally loads weights for IR-SE-50 backbone.
    Args:
        weights: Path to pretrained weights file
    Returns:
        Initialized IR-SE-50 model
    """
    net = BackboneIRSE50()
    if weights:
        sd = torch.load(weights, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        # Remove 'module.' prefix from state dict keys
        sd = {re.sub(r"^module\.", "", k): v for k, v in sd.items()}
        missing, _ = net.load_state_dict(sd, strict=False)
        print(
            f"[INFO] ArcFace's pretrained weights loaded. Missing: {len(missing)} params"
        )
    return net


##################################  MobileFaceNet #############################################################


class Conv_block(Module):
    def __init__(
        self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1
    ):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(
            in_c,
            out_channels=out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    def __init__(
        self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1
    ):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(
            in_c,
            out_channels=out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):
    def __init__(
        self,
        in_c,
        out_c,
        residual=False,
        kernel=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        groups=1,
    ):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(
            in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.conv_dw = Conv_block(
            groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride
        )
        self.project = Linear_block(
            groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(
        self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
    ):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Depth_Wise(
                    c,
                    c,
                    residual=True,
                    kernel=kernel,
                    padding=padding,
                    stride=stride,
                    groups=groups,
                )
            )
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MobileFaceNet(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(
            64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64
        )
        self.conv_23 = Depth_Wise(
            64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128
        )
        self.conv_3 = Residual(
            64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv_34 = Depth_Wise(
            64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256
        )
        self.conv_4 = Residual(
            128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv_45 = Depth_Wise(
            128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512
        )
        self.conv_5 = Residual(
            128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv_6_sep = Conv_block(
            128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.conv_6_dw = Linear_block(
            512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0)
        )
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_dw(out)

        out = self.conv_23(out)

        out = self.conv_3(out)

        out = self.conv_34(out)

        out = self.conv_4(out)

        out = self.conv_45(out)

        out = self.conv_5(out)

        out = self.conv_6_sep(out)

        out = self.conv_6_dw(out)

        out = self.conv_6_flatten(out)

        out = self.linear(out)

        out = self.bn(out)
        return l2_norm(out)
