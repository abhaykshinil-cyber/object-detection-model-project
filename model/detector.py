"""
Full detector: Backbone → FPN Neck → Multi-scale Head → Decode.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from model.backbone import Backbone
from model.neck import FPN
from model.head import MultiScaleHead

# Default anchors (w, h) in pixels at 640×640 input.
# These were derived via k-means on COCO bounding boxes.
# Run utils/anchors.py on your own dataset to get better anchors.
DEFAULT_ANCHORS: List[List[Tuple[int, int]]] = [
    [(10, 13),  (16, 30),   (33, 23)],    # P3 / stride 8  — small objects
    [(30, 61),  (62, 45),   (59, 119)],   # P4 / stride 16 — medium objects
    [(116, 90), (156, 198), (373, 326)],  # P5 / stride 32 — large objects
]

STRIDES: List[int] = [8, 16, 32]


class CustomDetector(nn.Module):
    """
    End-to-end object detector.

    Architecture:
        Backbone : Custom 5-stage CSP-DarkNet CNN
        Neck     : Feature Pyramid Network (FPN)
        Head     : YOLO-style multi-scale detection heads
        Loss     : CIoU + BCE objectness + BCE classification
        Decode   : Grid offset + anchor scaling → absolute pixel boxes
    """

    def __init__(
        self,
        num_classes: int = 80,
        anchors: Optional[List] = None,
        neck_out_ch: int = 64,
        decoupled: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors or DEFAULT_ANCHORS
        self.strides = STRIDES
        self.num_anchors = len(self.anchors[0])

        self.backbone = Backbone()
        self.neck = FPN(in_channels=(64, 128, 256), out_ch=neck_out_ch)
        self.head = MultiScaleHead(
            in_ch=neck_out_ch,
            num_anchors=self.num_anchors,
            num_classes=num_classes,
            decoupled=decoupled,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) — normalised to [0, 1]
        Returns:
            Tuple of 3 raw prediction tensors (one per scale).
            Each: (B, num_anchors*(5+num_classes), H_i, W_i)
        """
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c3, c4, c5)
        return self.head(p3, p4, p5)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        conf_thresh: float = 0.25,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference and return decoded + filtered boxes per image.

        Returns list (one entry per image in batch) of dicts:
            {"boxes": (N,4) x1y1x2y2, "scores": (N,), "classes": (N,) int}
        """
        self.eval()
        raw = self(x)
        return decode_predictions(raw, self.anchors, self.strides, self.num_classes, conf_thresh)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Decode raw predictions → absolute boxes
# ---------------------------------------------------------------------------

def decode_predictions(
    raw_preds: Tuple[torch.Tensor, ...],
    anchors: List[List[Tuple[int, int]]],
    strides: List[int],
    num_classes: int,
    conf_thresh: float = 0.25,
) -> List[Dict[str, torch.Tensor]]:
    """
    Decode YOLO-format raw head outputs into absolute pixel boxes.

    For each grid cell and anchor:
        cx = (sigmoid(tx) + grid_x) * stride
        cy = (sigmoid(ty) + grid_y) * stride
        bw = exp(tw) * anchor_w
        bh = exp(th) * anchor_h
        score = sigmoid(obj) * sigmoid(cls)

    Returns one dict per image in the batch.
    """
    batch_size = raw_preds[0].shape[0]
    all_boxes: List[torch.Tensor] = []
    all_scores: List[torch.Tensor] = []

    for pred, anc_list, stride in zip(raw_preds, anchors, strides):
        B, _, Gy, Gx = pred.shape
        na = len(anc_list)
        nc = num_classes

        # (B, na, Gy, Gx, 5+nc)
        pred = pred.view(B, na, 5 + nc, Gy, Gx).permute(0, 1, 3, 4, 2).contiguous()

        obj = torch.sigmoid(pred[..., 0])        # (B, na, Gy, Gx)
        tx  = torch.sigmoid(pred[..., 1])
        ty  = torch.sigmoid(pred[..., 2])
        tw  = pred[..., 3]
        th  = pred[..., 4]
        cls = torch.sigmoid(pred[..., 5:])       # (B, na, Gy, Gx, nc)

        # Build spatial grids
        gy, gx = torch.meshgrid(
            torch.arange(Gy, device=pred.device, dtype=pred.dtype),
            torch.arange(Gx, device=pred.device, dtype=pred.dtype),
            indexing="ij",
        )
        gx = gx[None, None]   # (1, 1, Gy, Gx)
        gy = gy[None, None]

        anc_t = torch.tensor(anc_list, device=pred.device, dtype=pred.dtype)
        aw = anc_t[:, 0].view(1, na, 1, 1)
        ah = anc_t[:, 1].view(1, na, 1, 1)

        # Decode to absolute pixel coordinates
        cx = (tx + gx) * stride
        cy = (ty + gy) * stride
        bw = torch.exp(tw.clamp(-4, 4)) * aw
        bh = torch.exp(th.clamp(-4, 4)) * ah

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        boxes  = torch.stack([x1, y1, x2, y2], dim=-1).view(B, -1, 4)
        scores = (obj.unsqueeze(-1) * cls).view(B, -1, nc)

        all_boxes.append(boxes)
        all_scores.append(scores)

    all_boxes  = torch.cat(all_boxes,  dim=1)  # (B, N_total, 4)
    all_scores = torch.cat(all_scores, dim=1)  # (B, N_total, nc)

    results = []
    for b in range(batch_size):
        max_scores, cls_ids = all_scores[b].max(dim=-1)
        mask = max_scores > conf_thresh
        results.append({
            "boxes":   all_boxes[b][mask],
            "scores":  max_scores[mask],
            "classes": cls_ids[mask],
        })
    return results
