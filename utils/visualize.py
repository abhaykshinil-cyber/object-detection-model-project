"""
Visualisation utilities — draw bounding boxes on images.
"""
import random
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Deterministic colour palette (one colour per class index)
_PALETTE = [
    (tuple(random.Random(i).randint(60, 230) for _ in range(3)))
    for i in range(200)
]


def draw_boxes(
    image: np.ndarray,
    boxes,
    scores,
    classes,
    class_names: Optional[List[str]] = None,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw detection boxes and labels on a copy of the image.

    Args:
        image:       (H, W, 3) uint8 RGB numpy array.
        boxes:       (N, 4) [x1, y1, x2, y2] — can be torch.Tensor or ndarray.
        scores:      (N,)   confidence values.
        classes:     (N,)   integer class indices.
        class_names: optional list mapping class index → name string.
        thickness:   box border thickness in pixels.

    Returns:
        (H, W, 3) uint8 annotated RGB image.
    """
    # Normalise to plain Python lists
    boxes   = _to_list(boxes)
    scores  = _to_list(scores)
    classes = _to_list(classes)

    pil = Image.fromarray(image.astype(np.uint8))
    draw = ImageDraw.Draw(pil)

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        # Skip degenerate boxes (can occur with an undertrained model)
        if x2 <= x1 or y2 <= y1:
            continue
        cls_int = int(cls)
        color = _PALETTE[cls_int % len(_PALETTE)]

        if class_names and cls_int < len(class_names):
            label = f"{class_names[cls_int]} {float(score):.2f}"
        else:
            label = f"cls{cls_int} {float(score):.2f}"

        # Draw box
        for t in range(thickness):
            draw.rectangle([x1 + t, y1 + t, x2 - t, y2 - t], outline=color)

        # Draw label background + text
        try:
            font = ImageFont.load_default()
            bbox = draw.textbbox((x1, y1), label, font=font)
        except AttributeError:
            # Older Pillow versions
            tw, th = len(label) * 6, 12
            bbox = (int(x1), int(y1) - th, int(x1) + tw, int(y1))

        draw.rectangle(bbox, fill=color)
        draw.text((bbox[0], bbox[1]), label, fill=(255, 255, 255))

    return np.array(pil)


def _to_list(x):
    try:
        import torch
        if torch.is_tensor(x):
            return x.cpu().numpy().tolist()
    except ImportError:
        pass
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)
