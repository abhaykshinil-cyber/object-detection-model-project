"""
Object detection evaluation metrics.

  compute_map()  — mAP@IoU (Pascal VOC or COCO-style) per class + mean.
  compute_iou_matrix()  — N×M pairwise IoU matrix (numpy).
"""
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch


# ---------------------------------------------------------------------------
# IoU matrix
# ---------------------------------------------------------------------------

def compute_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between predicted boxes and GT boxes.

    Args:
        pred_boxes: (N, 4) [x1, y1, x2, y2]
        gt_boxes:   (M, 4) [x1, y1, x2, y2]

    Returns:
        (N, M) IoU matrix.
    """
    area_p = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_g = (gt_boxes[:, 2]   - gt_boxes[:, 0])   * (gt_boxes[:, 3]   - gt_boxes[:, 1])

    ix1 = np.maximum(pred_boxes[:, None, 0], gt_boxes[None, :, 0])
    iy1 = np.maximum(pred_boxes[:, None, 1], gt_boxes[None, :, 1])
    ix2 = np.minimum(pred_boxes[:, None, 2], gt_boxes[None, :, 2])
    iy2 = np.minimum(pred_boxes[:, None, 3], gt_boxes[None, :, 3])

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    union = area_p[:, None] + area_g[None, :] - inter
    return inter / np.maximum(union, 1e-6)


# ---------------------------------------------------------------------------
# Average Precision
# ---------------------------------------------------------------------------

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Area under the Precision-Recall curve using 101-point COCO interpolation.
    """
    recalls    = np.concatenate([[0.0], recalls,    [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])
    # Monotonically decreasing precision envelope
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    return float(np.trapezoid(precisions, recalls))


# ---------------------------------------------------------------------------
# mAP
# ---------------------------------------------------------------------------

def compute_map(
    predictions: List[Dict],   # [{"boxes":(N,4), "scores":(N,), "classes":(N,)}, …]
    ground_truths: List[Dict], # [{"boxes":(M,4), "classes":(M,)}, …]
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute per-class Average Precision and mean AP.

    Matching rule:
        For each predicted box (sorted descending by score), match to the
        unmatched GT box with highest IoU ≥ iou_threshold.  Duplicate matches
        count as false positives.

    Returns:
        {
          "mAP":      float,
          "AP_cls0":  float,
          "AP_cls1":  float,
          ...
        }
    """
    cls_preds: Dict[int, List] = defaultdict(list)  # cls → [(score, is_tp)]
    cls_ngt:   Dict[int, int]  = defaultdict(int)   # cls → total GT count

    for pred, gt in zip(predictions, ground_truths):
        p_boxes   = _to_numpy(pred["boxes"])
        p_scores  = _to_numpy(pred["scores"])
        p_classes = _to_numpy(pred["classes"]).astype(int)
        g_boxes   = _to_numpy(gt["boxes"])
        g_classes = _to_numpy(gt["classes"]).astype(int)

        for cls in range(num_classes):
            gt_mask = g_classes == cls
            n_gt    = int(gt_mask.sum())
            cls_ngt[cls] += n_gt

            pd_mask = p_classes == cls
            if pd_mask.sum() == 0:
                continue

            pd_boxes  = p_boxes[pd_mask]
            pd_scores = p_scores[pd_mask]

            if n_gt == 0:
                for s in pd_scores:
                    cls_preds[cls].append((float(s), False))
                continue

            gt_cls_boxes = g_boxes[gt_mask]
            matched = np.zeros(n_gt, dtype=bool)
            iou_mat = compute_iou_matrix(pd_boxes, gt_cls_boxes)

            for det_idx in pd_scores.argsort()[::-1]:
                best_j   = int(iou_mat[det_idx].argmax())
                best_iou = float(iou_mat[det_idx, best_j])
                is_tp    = best_iou >= iou_threshold and not matched[best_j]
                if is_tp:
                    matched[best_j] = True
                cls_preds[cls].append((float(pd_scores[det_idx]), is_tp))

    aps: Dict[int, float] = {}
    for cls in range(num_classes):
        entries = sorted(cls_preds[cls], key=lambda e: -e[0])
        if not entries or cls_ngt[cls] == 0:
            aps[cls] = 0.0
            continue
        tp_cum = np.cumsum([e[1] for e in entries], dtype=float)
        fp_cum = np.cumsum([not e[1] for e in entries], dtype=float)
        recalls    = tp_cum / (cls_ngt[cls] + 1e-8)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
        aps[cls]   = compute_ap(recalls, precisions)

    map50 = float(np.mean(list(aps.values()))) if aps else 0.0
    return {"mAP": map50, **{f"AP_cls{k}": v for k, v in aps.items()}}


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return np.asarray(x)
