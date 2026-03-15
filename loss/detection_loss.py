"""
Combined detection loss (YOLO-style).

Components:
  - CIoU loss       for bounding-box regression (positive anchors only)
  - BCE loss        for objectness (all grid cells)
  - BCE loss        for class prediction (positive anchors only)

Target assignment:
  An anchor is "positive" for a GT box if the ratio between the anchor
  shape and the GT shape is below `anchor_thresh` in both dimensions
  (YOLOv5-style shape-based matching — no IoU computation at this stage).
"""
import torch
import torch.nn as nn

from loss.iou import bbox_iou


class DetectionLoss(nn.Module):
    def __init__(
        self,
        anchors,           # list of 3 lists, each [(w,h), (w,h), (w,h)]
        strides,           # [8, 16, 32]
        num_classes: int,
        lambda_box: float = 0.05,
        lambda_obj: float = 1.0,
        lambda_cls: float = 0.5,
        anchor_thresh: float = 4.0,
    ):
        super().__init__()
        self.anchors = anchors
        self.strides = strides
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.anchor_thresh = anchor_thresh
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        predictions,          # tuple of 3 raw tensors (B, na*(5+nc), Gy, Gx)
        targets: torch.Tensor,  # (N_gt, 6) [batch_idx, cls, cx, cy, w, h] normalised 0-1
        img_size: int,
    ):
        """
        Returns:
            total_loss (scalar), component_dict {"box", "obj", "cls"}
        """
        device = predictions[0].device
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        loss_cls = torch.zeros(1, device=device)

        for pred, anc_list, stride in zip(predictions, self.anchors, self.strides):
            B, _, Gy, Gx = pred.shape
            na = len(anc_list)
            nc = self.num_classes

            # Reshape: (B, na, Gy, Gx, 5+nc)
            pred = pred.view(B, na, 5 + nc, Gy, Gx).permute(0, 1, 3, 4, 2).contiguous()

            # Objectness target (all cells, default 0)
            t_obj = torch.zeros(B, na, Gy, Gx, device=device)

            p_boxes,  t_boxes  = [], []
            p_cls_lg, t_cls_oh = [], []

            if targets.shape[0] > 0:
                anc_t = torch.tensor(anc_list, dtype=torch.float32, device=device)  # (na, 2)

                # GT width/height in absolute pixels
                gt_wh = targets[:, 4:6] * img_size  # (N_gt, 2)

                for anc_idx in range(na):
                    aw, ah = anc_list[anc_idx]
                    # Anchor/GT ratio test — keep GTs whose shape is "close" to this anchor
                    r = gt_wh / torch.tensor([[aw, ah]], dtype=torch.float32, device=device)
                    keep = torch.max(r, 1.0 / r.clamp(min=1e-8)).max(dim=1).values < self.anchor_thresh
                    tgt = targets[keep]            # (M, 6)
                    if tgt.shape[0] == 0:
                        continue

                    bi  = tgt[:, 0].long()                  # batch index
                    cls = tgt[:, 1].long()                  # class index
                    cx  = tgt[:, 2] * Gx                    # centre x in grid coords
                    cy  = tgt[:, 3] * Gy                    # centre y in grid coords
                    gw  = tgt[:, 4] * Gx                    # width  in grid coords
                    gh  = tgt[:, 5] * Gy                    # height in grid coords

                    gi = cx.long().clamp(0, Gx - 1)        # grid cell column
                    gj = cy.long().clamp(0, Gy - 1)        # grid cell row

                    # Mark positive cells
                    t_obj[bi, anc_idx, gj, gi] = 1.0

                    # --- Predicted box at positive cells ---
                    p = pred[bi, anc_idx, gj, gi]           # (M, 5+nc)
                    p_cx = torch.sigmoid(p[:, 1]) + gi.float()
                    p_cy = torch.sigmoid(p[:, 2]) + gj.float()
                    p_bw = torch.exp(p[:, 3].clamp(-4, 4)) * aw / stride
                    p_bh = torch.exp(p[:, 4].clamp(-4, 4)) * ah / stride

                    # Convert to pixel-space x1y1x2y2
                    pb = torch.stack([
                        (p_cx - p_bw / 2) * stride,
                        (p_cy - p_bh / 2) * stride,
                        (p_cx + p_bw / 2) * stride,
                        (p_cy + p_bh / 2) * stride,
                    ], dim=-1)

                    # --- Ground-truth box (pixel-space x1y1x2y2) ---
                    tb = torch.stack([
                        (cx - gw / 2) * stride,
                        (cy - gh / 2) * stride,
                        (cx + gw / 2) * stride,
                        (cy + gh / 2) * stride,
                    ], dim=-1)

                    p_boxes.append(pb)
                    t_boxes.append(tb)

                    # Class logits + one-hot targets
                    p_cls_lg.append(p[:, 5:])
                    one_hot = torch.zeros(len(cls), nc, device=device)
                    one_hot[torch.arange(len(cls)), cls] = 1.0
                    t_cls_oh.append(one_hot)

            # Objectness loss (all cells)
            loss_obj = loss_obj + self.bce(pred[..., 0], t_obj)

            # Box regression loss (positive cells only)
            if p_boxes:
                ciou = bbox_iou(torch.cat(p_boxes), torch.cat(t_boxes), mode="ciou")
                loss_box = loss_box + (1.0 - ciou).mean()

            # Classification loss (positive cells only)
            if p_cls_lg:
                loss_cls = loss_cls + self.bce(
                    torch.cat(p_cls_lg), torch.cat(t_cls_oh)
                )

        total = (
            self.lambda_box * loss_box
            + self.lambda_obj * loss_obj
            + self.lambda_cls * loss_cls
        )
        return total, {
            "box": loss_box.item(),
            "obj": loss_obj.item(),
            "cls": loss_cls.item(),
        }
