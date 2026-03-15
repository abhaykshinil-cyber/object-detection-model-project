"""
Non-Maximum Suppression — pure PyTorch, no external dependencies.

Functions:
  nms()              — standard single-class NMS
  multiclass_nms()   — per-class NMS, then merge and keep top-K
"""
import torch


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.45) -> torch.Tensor:
    """
    Standard greedy NMS.

    Args:
        boxes:         (N, 4)  [x1, y1, x2, y2]
        scores:        (N,)    confidence scores
        iou_threshold: suppress boxes with IoU > this value

    Returns:
        (K,) LongTensor of kept indices, sorted by descending score.
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    order = scores.argsort(descending=True)
    kept: list = []

    while order.numel() > 0:
        i = order[0].item()
        kept.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        ix1 = x1[rest].clamp(min=x1[i].item())
        iy1 = y1[rest].clamp(min=y1[i].item())
        ix2 = x2[rest].clamp(max=x2[i].item())
        iy2 = y2[rest].clamp(max=y2[i].item())

        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        union = areas[i] + areas[rest] - inter
        iou = inter / union.clamp(min=1e-6)

        order = rest[iou <= iou_threshold]

    return torch.tensor(kept, dtype=torch.long, device=boxes.device)


def multiclass_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float = 0.45,
    max_det: int = 300,
) -> tuple:
    """
    Per-class NMS followed by a global top-K filter.

    Args:
        boxes:         (N, 4)  [x1, y1, x2, y2]
        scores:        (N,)
        classes:       (N,)  integer class ids
        iou_threshold: IoU threshold for suppression
        max_det:       maximum detections to return

    Returns:
        (boxes, scores, classes) — each a 1-D / 2-D tensor, potentially empty.
    """
    if boxes.numel() == 0:
        empty = torch.zeros(0, device=boxes.device)
        return torch.zeros((0, 4), device=boxes.device), empty, empty.long()

    kept_boxes, kept_scores, kept_classes = [], [], []

    for cls_id in classes.unique():
        mask = classes == cls_id
        idx = nms(boxes[mask], scores[mask], iou_threshold)
        kept_boxes.append(boxes[mask][idx])
        kept_scores.append(scores[mask][idx])
        kept_classes.append(classes[mask][idx])

    all_boxes   = torch.cat(kept_boxes)
    all_scores  = torch.cat(kept_scores)
    all_classes = torch.cat(kept_classes)

    if all_scores.shape[0] > max_det:
        top_idx = all_scores.topk(max_det).indices
        all_boxes, all_scores, all_classes = (
            all_boxes[top_idx], all_scores[top_idx], all_classes[top_idx]
        )

    return all_boxes, all_scores, all_classes
