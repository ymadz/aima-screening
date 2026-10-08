"""Compact MobileNetV3-Small-encoder U-Net for binary anatomical masks."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((x, skip), dim=1))


class MobileNetSmallUNet(nn.Module):
    """U-Net-like decoder with frozen architecture, optionally pretrained encoder."""

    def __init__(self, pretrained_encoder: bool = True) -> None:
        super().__init__()
        weights = None
        if pretrained_encoder:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(
                urlparse(weights.url).path
            ).name
            if not cached.is_file():
                raise FileNotFoundError(
                    f"Pretrained encoder is not cached at {cached}; "
                    "explicitly download it or use --no-pretrained-encoder"
                )
        self.encoder = mobilenet_v3_small(weights=weights).features
        self.up_16 = UpBlock(576, 48, 128)
        self.up_32 = UpBlock(128, 24, 64)
        self.up_64 = UpBlock(64, 16, 32)
        self.up_128 = UpBlock(32, 16, 16)
        self.final = ConvBlock(16, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        skips: dict[int, torch.Tensor] = {}
        x = image
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index in (0, 1, 3, 8):
                skips[index] = x
        x = self.up_16(x, skips[8])
        x = self.up_32(x, skips[3])
        x = self.up_64(x, skips[1])
        x = self.up_128(x, skips[0])
        x = F.interpolate(x, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.final(x))


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Balanced BCE-with-logits and soft Dice loss for sparse masks."""
    if logits.shape != target.shape:
        raise ValueError("Logits and target mask shapes must match")
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    soft_dice = (2 * intersection + 1.0) / (denominator + 1.0)
    return 0.5 * bce + 0.5 * (1.0 - soft_dice.mean())
