"""
FPN (Feature Pyramid Network) neck.

Takes (C3, C4, C5) from the backbone and produces (P3, P4, P5):
  - Top-down lateral merging for rich multi-scale features.
  - All output scales share the same channel width (out_ch).
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import ConvBN, CSPBlock


class FPN(nn.Module):
    """
    Top-down FPN with lateral connections and CSP refinement.

    Input : C3 (B, 256, H/8,  W/8)
            C4 (B, 512, H/16, W/16)
            C5 (B,1024, H/32, W/32)
    Output: P3 (B, out_ch, H/8,  W/8)   ← small objects
            P4 (B, out_ch, H/16, W/16)  ← medium objects
            P5 (B, out_ch, H/32, W/32)  ← large objects
    """

    def __init__(self, in_channels: Tuple = (256, 512, 1024), out_ch: int = 256):
        super().__init__()
        c3_ch, c4_ch, c5_ch = in_channels

        # Lateral 1×1 projections (reduce backbone channels to out_ch)
        self.lat5 = ConvBN(c5_ch, out_ch, k=1)
        self.lat4 = ConvBN(c4_ch, out_ch, k=1)
        self.lat3 = ConvBN(c3_ch, out_ch, k=1)

        # After concatenation with upsampled feature: 2*out_ch → out_ch
        self.merge4 = CSPBlock(out_ch * 2, out_ch, n=1)
        self.merge3 = CSPBlock(out_ch * 2, out_ch, n=1)

        # P5 refinement (no concat needed at top level)
        self.refine5 = CSPBlock(out_ch, out_ch, n=1)

    def forward(
        self,
        c3: torch.Tensor,
        c4: torch.Tensor,
        c5: torch.Tensor,
    ):
        # Top level
        p5 = self.refine5(self.lat5(c5))                               # (B, out_ch, H/32, W/32)

        # Merge into P4
        p5_up = F.interpolate(p5, scale_factor=2, mode="nearest")
        p4 = self.merge4(torch.cat([self.lat4(c4), p5_up], dim=1))    # (B, out_ch, H/16, W/16)

        # Merge into P3
        p4_up = F.interpolate(p4, scale_factor=2, mode="nearest")
        p3 = self.merge3(torch.cat([self.lat3(c3), p4_up], dim=1))    # (B, out_ch, H/8,  W/8)

        return p3, p4, p5
