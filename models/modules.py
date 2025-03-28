import torch
import torch.nn as nn
import torch.nn.functional as F
from .binarization import DualRateAlphaFunction

class BinaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        binary_weight = DualRateAlphaFunction.apply(self.conv.weight, self.alpha)
        x = F.conv2d(x, binary_weight, stride=self.conv.stride, padding=self.conv.padding)
        x = self.bn(x)
        return self.prelu(x)

class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        binary_weight = DualRateAlphaFunction.apply(self.fc.weight, self.alpha)
        return F.linear(x, binary_weight)

class BinaryResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = BinaryConv2d(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = BinaryConv2d(out_channels, out_channels, kernel_size=3, stride=1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                BinaryConv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        return out