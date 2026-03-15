"""
Evaluate a trained checkpoint.

Usage:
    # Custom dataset
    python eval.py --checkpoint runs/train/best.pt \
                   --img-dir data/images/val \
                   --label-dir data/labels/val

    # COCO
    python eval.py --checkpoint runs/train/best.pt \
                   --data-type coco --data-root /data/coco --split val2017
"""
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import COCODetectionDataset, CustomDetectionDataset, collate_fn
from model.detector import CustomDetector
from utils.metrics import compute_map
from utils.nms import multiclass_nms


def evaluate(checkpoint_path: str, cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt["num_classes"]
    anchors     = ckpt["anchors"]
    class_names = ckpt.get("class_names", [str(i) for i in range(num_classes)])
    img_size    = ckpt.get("img_size", cfg.get("img_size", 640))

    model = CustomDetector(num_classes=num_classes, anchors=anchors).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if cfg["data_type"] == "coco":
        dataset = COCODetectionDataset(
            root=cfg["data_root"],
            split=cfg.get("split", "val2017"),
            img_size=img_size,
            augment=False,
        )
    else:
        dataset = CustomDetectionDataset(
            img_dir=cfg["img_dir"],
            label_dir=cfg["label_dir"],
            class_names=class_names,
            img_size=img_size,
            augment=False,
        )

    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 8),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    all_preds, all_gts = [], []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images  = images.to(device)
            B       = images.shape[0]

            batch_results = model.predict(images, conf_thresh=cfg.get("conf_thresh", 0.25))

            for b in range(B):
                res = batch_results[b]
                boxes, scores, classes = multiclass_nms(
                    res["boxes"], res["scores"], res["classes"],
                    iou_threshold=cfg.get("nms_iou", 0.45),
                )
                all_preds.append({
                    "boxes":   boxes.cpu(),
                    "scores":  scores.cpu(),
                    "classes": classes.cpu(),
                })

                # Extract GT for this image
                mask = (targets[:, 0] == b)
                gt_t = targets[mask]
                if gt_t.shape[0] > 0:
                    cls_gt = gt_t[:, 1].long()
                    cx  = gt_t[:, 2] * img_size
                    cy  = gt_t[:, 3] * img_size
                    w   = gt_t[:, 4] * img_size
                    h   = gt_t[:, 5] * img_size
                    gt_boxes = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)
                else:
                    gt_boxes = torch.zeros((0, 4))
                    cls_gt   = torch.zeros(0, dtype=torch.long)

                all_gts.append({"boxes": gt_boxes, "classes": cls_gt})

    metrics = compute_map(
        all_preds, all_gts, num_classes,
        iou_threshold=cfg.get("iou_threshold", 0.5),
    )

    print(f"\nmAP@{cfg.get('iou_threshold', 0.5):.2f}: {metrics['mAP']:.4f}\n")
    for i, name in enumerate(class_names):
        ap = metrics.get(f"AP_cls{i}", 0.0)
        print(f"  {name:<20s}: {ap:.4f}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate object detector checkpoint")
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--data-type",     choices=["coco", "custom"], default="custom")
    parser.add_argument("--data-root",     help="COCO root")
    parser.add_argument("--split",         default="val2017")
    parser.add_argument("--img-dir")
    parser.add_argument("--label-dir")
    parser.add_argument("--img-size",      type=int, default=640)
    parser.add_argument("--batch-size",    type=int, default=8)
    parser.add_argument("--conf-thresh",   type=float, default=0.25)
    parser.add_argument("--nms-iou",       type=float, default=0.45)
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for TP/FP matching (default 0.5 = mAP@0.5)")
    args = parser.parse_args()

    cfg = {k.replace("-", "_"): v for k, v in vars(args).items()}
    evaluate(args.checkpoint, cfg)


if __name__ == "__main__":
    main()
