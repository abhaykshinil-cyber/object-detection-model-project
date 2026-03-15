"""
Detection heads — two implementations:

  CoupledHead     : original shared-tower YOLO-style head (coupled cls + reg)
  DecoupledHead   : YOLOX-style separate classification and regression branches
                    (default — better mAP, especially on crowded scenes)

The decoupled design avoids the conflict between the classification task
(wants high-level semantic features) and the localisation task (wants
low-level spatial features) that hurts coupled heads.
"""
import torch
import torch.nn as nn

from model.backbone import ConvBN


# ---------------------------------------------------------------------------
# Coupled head (original)
# ---------------------------------------------------------------------------

class CoupledHead(nn.Module):
    """Single shared 5-conv tower → predicts obj + reg + cls together."""

    def __init__(self, in_ch: int, num_anchors: int, num_classes: int):
        super().__init__()
        hidden = in_ch * 2
        self.tower = nn.Sequential(
            ConvBN(in_ch,  hidden, k=3),
            ConvBN(hidden, in_ch,  k=1),
            ConvBN(in_ch,  hidden, k=3),
            ConvBN(hidden, in_ch,  k=1),
            ConvBN(in_ch,  hidden, k=3),
        )
        self.pred = nn.Conv2d(hidden, num_anchors * (5 + num_classes), kernel_size=1)
        nn.init.constant_(self.pred.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred(self.tower(x))


# ---------------------------------------------------------------------------
# Decoupled head (YOLOX-style)
# ---------------------------------------------------------------------------

class DecoupledHead(nn.Module):
    """
    Separate branches for:
      - Classification  (2-conv tower → cls logits)
      - Regression + Objectness (2-conv tower → box + obj logits)

    Output layout matches CoupledHead:
      (B, num_anchors * (5 + num_classes), H, W)
      where 5 = [obj, tx, ty, tw, th]

    Why decoupled:
      Classification needs high-level semantic context.
      Regression needs precise spatial detail.
      Sharing the same feature map for both forces a suboptimal compromise
      (+2–3 mAP on COCO vs the coupled head).
    """

    def __init__(self, in_ch: int, num_anchors: int, num_classes: int):
        super().__init__()
        self.na = num_anchors
        self.nc = num_classes

        # Shared stem reduces channels before splitting
        self.stem = ConvBN(in_ch, in_ch, k=1)

        # Classification branch
        self.cls_branch = nn.Sequential(
            ConvBN(in_ch, in_ch, k=3),
            ConvBN(in_ch, in_ch, k=3),
        )
        self.cls_pred = nn.Conv2d(in_ch, num_anchors * num_classes, kernel_size=1)

        # Regression + objectness branch
        self.reg_branch = nn.Sequential(
            ConvBN(in_ch, in_ch, k=3),
            ConvBN(in_ch, in_ch, k=3),
        )
        self.reg_pred = nn.Conv2d(in_ch, num_anchors * 4, kernel_size=1)   # tx, ty, tw, th
        self.obj_pred = nn.Conv2d(in_ch, num_anchors * 1, kernel_size=1)   # objectness

        # Bias init: small negative bias on cls/obj to stabilise first epochs
        nn.init.constant_(self.cls_pred.bias, -2.0)
        nn.init.constant_(self.obj_pred.bias, -2.0)
        nn.init.constant_(self.reg_pred.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)

        # Classification
        cls_feat = self.cls_branch(feat)
        cls_out  = self.cls_pred(cls_feat)          # (B, na*nc, H, W)

        # Regression + objectness
        reg_feat = self.reg_branch(feat)
        reg_out  = self.reg_pred(reg_feat)          # (B, na*4,  H, W)
        obj_out  = self.obj_pred(reg_feat)          # (B, na*1,  H, W)

        B, _, H, W = x.shape
        na, nc = self.na, self.nc

        # Reshape each to (B, na, H, W, *) then interleave into YOLO layout
        obj = obj_out.view(B, na, 1,  H, W)
        reg = reg_out.view(B, na, 4,  H, W)
        cls = cls_out.view(B, na, nc, H, W)

        # Concatenate: [obj, tx, ty, tw, th, cls...] → (B, na*(5+nc), H, W)
        out = torch.cat([obj, reg, cls], dim=2)     # (B, na, 5+nc, H, W)
        return out.view(B, na * (5 + nc), H, W)


# ---------------------------------------------------------------------------
# Multi-scale head wrapper
# ---------------------------------------------------------------------------

class MultiScaleHead(nn.Module):
    """
    Three detection heads (one per FPN scale).

    Args:
        in_ch:       FPN output channels (default 256).
        num_anchors: Anchors per cell per scale (default 3).
        num_classes: Number of object classes.
        decoupled:   Use DecoupledHead (True) or CoupledHead (False).
    """

    def __init__(
        self,
        in_ch: int = 256,
        num_anchors: int = 3,
        num_classes: int = 80,
        decoupled: bool = True,
    ):
        super().__init__()
        Head = DecoupledHead if decoupled else CoupledHead
        self.head_small  = Head(in_ch, num_anchors, num_classes)  # P3
        self.head_medium = Head(in_ch, num_anchors, num_classes)  # P4
        self.head_large  = Head(in_ch, num_anchors, num_classes)  # P5

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        return self.head_small(p3), self.head_medium(p4), self.head_large(p5)
