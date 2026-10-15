"""Deliberately small, deterministic PyTorch models for native EE experiments.

These models use only inference operations that have raster equivalents.  They
are random numerical fixtures, not trained segmentation models.

Formerly ``tensor_ee/models.py`` (decision D3); ``tensor_ee.models`` re-exports
every name, so ``from tensor_ee.models import TinyUNet`` keeps working.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 3, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.head = nn.Conv2d(3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.sigmoid(self.head(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        return self.relu2(self.conv2(self.relu1(self.conv1(x))))


class TinyUNet(nn.Module):
    """One or two pooling levels, nearest upsampling, and genuine skip concat."""

    def __init__(self, in_channels: int = 2, base_channels: int = 4,
                 levels: int = 1, pool: str = "average"):
        super().__init__()
        if levels not in (1, 2):
            raise ValueError("TinyUNet supports one or two levels")
        if pool not in ("average", "max"):
            raise ValueError("pool must be 'average' or 'max'")
        self.levels = levels
        self.enc1 = ConvBlock(in_channels, base_channels)
        pool_type = nn.AvgPool2d if pool == "average" else nn.MaxPool2d
        self.pool1 = pool_type(2, stride=2)
        if levels == 2:
            self.enc2 = ConvBlock(base_channels, base_channels * 2)
            self.pool2 = pool_type(2, stride=2)
            self.bottom = ConvBlock(base_channels * 2, base_channels * 4)
            self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
            self.dec2 = ConvBlock(base_channels * 6, base_channels * 2)
        else:
            self.bottom = ConvBlock(base_channels, base_channels * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = ConvBlock(base_channels * 3, base_channels)
        self.head = nn.Conv2d(base_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip1 = self.enc1(x)
        x = self.pool1(skip1)
        if self.levels == 2:
            skip2 = self.enc2(x)
            x = self.bottom(self.pool2(skip2))
            x = self.dec2(torch.cat([self.up2(x), skip2], dim=1))
        else:
            x = self.bottom(x)
        x = self.dec1(torch.cat([self.up1(x), skip1], dim=1))
        return self.sigmoid(self.head(x))


class PoolRoundTrip(nn.Module):
    def __init__(self, pool: str = "average"):
        super().__init__()
        if pool not in ("average", "max"):
            raise ValueError("pool must be 'average' or 'max'")
        self.pool = (nn.AvgPool2d if pool == "average" else nn.MaxPool2d)(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        return self.up(self.pool(x))


def initialize_fixture(model: nn.Module, seed: int = 1729) -> nn.Module:
    """Use reproducible nonsymmetric weights with measurable spatial variation.

    A local Generator avoids modifying callers' global torch random state.
    The fixtures deliberately include negative weights and nonzero biases.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                fan_in = module.in_channels * math.prod(module.kernel_size)
                module.weight.copy_(torch.randn(module.weight.shape,
                                               generator=generator) *
                                    math.sqrt(2.0 / fan_in) * 0.85)
                if module.bias is not None:
                    module.bias.copy_(torch.randn(module.bias.shape,
                                                 generator=generator) * 0.05)
    return model.eval()


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
