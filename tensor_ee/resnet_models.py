"""ResNet18 U-Net numerical fixtures for native Earth Engine inference.

The standard-width encoder matches torchvision's ResNet18 through ``layer4``
with the classifier removed and an optional two-channel input convolution.
All ReLUs and residual additions are out of place, preserving the inference
function while allowing unambiguous FX capture. No pretrained weights are used.
The decoder is an explicit, conventional U-Net decoder rather than a claim of
matching any particular third-party U-Net package.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class BasicBlock(nn.Module):
    """Two 3x3 convolutions and the ResNet18 identity/projection residual."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18Encoder(nn.Module):
    """Canonical [2,2,2,2] BasicBlocks, 7x7 stem, and 3x3 padded max pool.

    ``base_channels=64`` gives widths 64,128,256,512. Smaller values retain
    block structure and spatial operators but are explicitly width-scaled
    ResNet18-style encoders. The returned skips have strides 2,4,8,16,32.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 64):
        super().__init__()
        if not isinstance(base_channels, int) or base_channels < 4:
            raise ValueError("base_channels must be an integer of at least four")
        if not isinstance(in_channels, int) or in_channels < 1:
            raise ValueError("in_channels must be a positive integer")
        self.base_channels = base_channels
        self.conv1 = nn.Conv2d(in_channels, base_channels, 7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(BasicBlock(base_channels, base_channels),
                                    BasicBlock(base_channels, base_channels))
        self.layer2 = nn.Sequential(BasicBlock(base_channels, base_channels * 2, 2),
                                    BasicBlock(base_channels * 2, base_channels * 2))
        self.layer3 = nn.Sequential(BasicBlock(base_channels * 2, base_channels * 4, 2),
                                    BasicBlock(base_channels * 4, base_channels * 4))
        self.layer4 = nn.Sequential(BasicBlock(base_channels * 4, base_channels * 8, 2),
                                    BasicBlock(base_channels * 8, base_channels * 8))

    def forward(self, x):
        stem = self.relu(self.bn1(self.conv1(x)))
        layer1 = self.layer1(self.maxpool(stem))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return stem, layer1, layer2, layer3, layer4


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu2(self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x))))))


class ResNet18UNet(nn.Module):
    """ResNet18 encoder + five nearest-neighbor decoder stages + 1x1 head.

    Standard decoder widths are 256,128,64,32,16 with four skip concats from
    encoder strides 16,8,4,2. The final stage upsamples to input resolution.
    Input height and width should be divisible by 32 for this fixture.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 64):
        super().__init__()
        if base_channels % 4:
            raise ValueError("base_channels must be divisible by four")
        self.base_channels = base_channels
        self.in_channels = in_channels
        self.encoder = ResNet18Encoder(in_channels, base_channels)
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec4 = DecoderBlock(base_channels * 12, base_channels * 4)
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = DecoderBlock(base_channels * 6, base_channels * 2)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = DecoderBlock(base_channels * 3, base_channels)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = DecoderBlock(base_channels * 2, base_channels // 2)
        self.up0 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec0 = DecoderBlock(base_channels // 2, base_channels // 4)
        self.head = nn.Conv2d(base_channels // 4, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        stem, layer1, layer2, layer3, layer4 = self.encoder(x)
        x = self.dec4(torch.cat([self.up4(layer4), layer3], dim=1))
        x = self.dec3(torch.cat([self.up3(x), layer2], dim=1))
        x = self.dec2(torch.cat([self.up2(x), layer1], dim=1))
        x = self.dec1(torch.cat([self.up1(x), stem], dim=1))
        x = self.dec0(self.up0(x))
        return self.sigmoid(self.head(x))


def initialize_resnet_fixture(model: nn.Module, seed: int = 1800) -> nn.Module:
    """Deterministic signed weights and nontrivial inference BatchNorm state.

    Nonzero running means, non-unit variance, gamma and beta make BatchNorm
    folding an actual numerical test. Modest gains keep logits unsaturated.
    The generator is local and fixtures do not change global RNG state.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                fan_in = module.in_channels * math.prod(module.kernel_size)
                module.weight.copy_(torch.randn(module.weight.shape,
                                                 generator=generator)
                                    * math.sqrt(2.0 / fan_in) * 0.75)
                if module.bias is not None:
                    module.bias.copy_(torch.randn(module.bias.shape,
                                                   generator=generator) * 0.05)
            elif isinstance(module, nn.BatchNorm2d):
                module.running_mean.copy_(torch.randn(module.running_mean.shape,
                                                       generator=generator) * 0.04)
                module.running_var.copy_(torch.rand(module.running_var.shape,
                                                      generator=generator) * 0.5 + 0.75)
                module.weight.copy_(torch.rand(module.weight.shape,
                                               generator=generator) * 0.3 + 0.8)
                module.bias.copy_(torch.randn(module.bias.shape,
                                               generator=generator) * 0.03)
                module.num_batches_tracked.fill_(7)
    return model.eval()


def fixture_metadata(model: ResNet18UNet, seed: int = 1800) -> dict:
    """JSON-compatible recipe suitable for weight-asset manifests."""
    base = model.base_channels
    return {
        "architecture": "ResNet18UNet",
        "encoder_kind": "standard_width_resnet18" if base == 64 else "width_scaled_resnet18_style",
        "base_channels": base,
        "in_channels": model.in_channels,
        "encoder_widths": [base, base * 2, base * 4, base * 8],
        "encoder_blocks": [2, 2, 2, 2],
        "decoder_widths": [base * 4, base * 2, base, base // 2, base // 4],
        "seed": seed,
        "fixture_initializer": "initialize_resnet_fixture_v1",
        "pretrained": False,
    }
