"""
IoU family implementations — all computed element-wise.

Supported modes: "iou" | "giou" | "diou" | "ciou"

CIoU adds a penalty for aspect-ratio inconsistency on top of DIoU,
which itself adds a centre-distance penalty on top of standard IoU.
"""
import math

import torch


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    eps: float = 1e-7,
    mode: str = "ciou",
) -> torch.Tensor:
    """
    Element-wise IoU (or variant) between two sets of boxes.

    Args:
        box1: (N, 4)  [x1, y1, x2, y2]
        box2: (N, 4)  [x1, y1, x2, y2]
        eps:  small constant for numerical stability
        mode: "iou" | "giou" | "diou" | "ciou"

    Returns:
        (N,) tensor — IoU values (higher = better overlap).
        Loss = 1 - bbox_iou(...)
    """
    b1x1, b1y1, b1x2, b1y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2x1, b2y1, b2x2, b2y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection
    ix1 = torch.max(b1x1, b2x1)
    iy1 = torch.max(b1y1, b2y1)
    ix2 = torch.min(b1x2, b2x2)
    iy2 = torch.min(b1y2, b2y2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    # Individual areas
    w1 = (b1x2 - b1x1).clamp(min=0)
    h1 = (b1y2 - b1y1).clamp(min=0)
    w2 = (b2x2 - b2x1).clamp(min=0)
    h2 = (b2y2 - b2y1).clamp(min=0)

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if mode == "iou":
        return iou

    # Smallest enclosing box
    enc_x1 = torch.min(b1x1, b2x1)
    enc_y1 = torch.min(b1y1, b2y1)
    enc_x2 = torch.max(b1x2, b2x2)
    enc_y2 = torch.max(b1y2, b2y2)
    enc_w  = (enc_x2 - enc_x1).clamp(min=0)
    enc_h  = (enc_y2 - enc_y1).clamp(min=0)

    if mode == "giou":
        enc_area = enc_w * enc_h + eps
        return iou - (enc_area - union) / enc_area

    # Centre-distance²
    cx1, cy1 = (b1x1 + b1x2) / 2, (b1y1 + b1y2) / 2
    cx2, cy2 = (b2x1 + b2x2) / 2, (b2y1 + b2y2) / 2
    rho2  = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    diag2 = enc_w ** 2 + enc_h ** 2 + eps

    if mode == "diou":
        return iou - rho2 / diag2

    # CIoU: aspect-ratio consistency term
    v = (4 / math.pi ** 2) * (
        torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))
    ) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return iou - rho2 / diag2 - alpha * v
