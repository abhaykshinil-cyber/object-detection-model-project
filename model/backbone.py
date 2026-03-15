"""
Custom CNN backbone — CSP-DarkNet style.

Outputs three feature maps:
  C3  (stride 8)  — small-object features    (256 ch)
  C4  (stride 16) — medium-object features   (512 ch)
  C5  (stride 32) — large-object features   (1024 ch)
"""
from typing import Tuple

import torch
import torch.nn as nn


def autopad(k: int) -> int:
    """Same-padding for stride-1 convolutions."""
    return k // 2


class ConvBN(nn.Module):
    """Conv2d → BatchNorm2d → LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, autopad(k), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """1×1 bottleneck → 3×3 → residual add."""

    def __init__(self, ch: int):
        super().__init__()
        hidden = ch // 2
        self.block = nn.Sequential(
            ConvBN(ch, hidden, k=1),
            ConvBN(hidden, ch, k=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CSPBlock(nn.Module):
    """
    Cross-Stage Partial block.
    Splits feature channels in two halves, applies residual stack to one half,
    then concatenates and projects back.
    """

    def __init__(self, in_ch: int, out_ch: int, n: int = 1):
        super().__init__()
        half = out_ch // 2
        self.conv_in = ConvBN(in_ch, out_ch, k=1)
        self.residuals = nn.Sequential(*[ResidualBlock(half) for _ in range(n)])
        self.conv_out = ConvBN(out_ch, out_ch, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        half = x.shape[1] // 2
        x1, x2 = x[:, :half], x[:, half:]
        x1 = self.residuals(x1)
        return self.conv_out(torch.cat([x1, x2], dim=1))


class Backbone(nn.Module):
    """
    5-stage custom backbone — micro width (0.25× channels vs full model).

    Input : (B, 3,  H,   W)
    Output: C3 (B, 64,  H/8,  W/8 )
            C4 (B, 128, H/16, W/16)
            C5 (B, 256, H/32, W/32)
    """

    def __init__(self):
        super().__init__()
        self.stem = ConvBN(3, 8, k=3, s=1)

        self.stage1 = nn.Sequential(
            ConvBN(8, 16, k=3, s=2),
            CSPBlock(16, 16, n=1),
        )

        self.stage2 = nn.Sequential(
            ConvBN(16, 32, k=3, s=2),
            CSPBlock(32, 32, n=1),
        )

        # C3 (stride 8)
        self.stage3 = nn.Sequential(
            ConvBN(32, 64, k=3, s=2),
            CSPBlock(64, 64, n=1),
        )

        # C4 (stride 16)
        self.stage4 = nn.Sequential(
            ConvBN(64, 128, k=3, s=2),
            CSPBlock(128, 128, n=1),
        )

        # C5 (stride 32)
        self.stage5 = nn.Sequential(
            ConvBN(128, 256, k=3, s=2),
            CSPBlock(256, 256, n=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        c3 = self.stage3(x)  # (B, 64,  H/8,  W/8)
        c4 = self.stage4(c3) # (B, 128, H/16, W/16)
        c5 = self.stage5(c4) # (B, 256, H/32, W/32)
        return c3, c4, c5
