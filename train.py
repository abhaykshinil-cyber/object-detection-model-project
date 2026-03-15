"""
Training script for the custom object detector.

Usage — custom dataset:
    python train.py --config configs/default.yaml

Usage — override via CLI:
    python train.py --data-type custom \
                    --img-dir data/images/train \
                    --label-dir data/labels/train \
                    --class-names person car bicycle \
                    --epochs 100 --batch-size 16

Usage — COCO:
    python train.py --data-type coco --data-root /data/coco --split train2017
"""
import argparse
import math
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from data.dataset import COCODetectionDataset, CustomDetectionDataset, collate_fn
from loss.detection_loss import DetectionLoss
from model.detector import CustomDetector, DEFAULT_ANCHORS, STRIDES


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_lr_lambda(epoch: int, total_epochs: int, warmup_epochs: int = 3) -> float:
    """Linear warm-up then cosine decay to 0.01× the initial LR."""
    if epoch < warmup_epochs:
        return max(epoch / max(warmup_epochs, 1), 1e-3)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return 0.01 + 0.5 * (1 - 0.01) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    device = torch.device(
        cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ---- Dataset ----
    mosaic_prob = cfg.get("mosaic_prob", 1.0)
    mixup_prob  = cfg.get("mixup_prob",  0.15)

    if cfg["data_type"] == "coco":
        dataset = COCODetectionDataset(
            root=cfg["data_root"],
            split=cfg.get("split", "train2017"),
            img_size=cfg["img_size"],
            augment=True,
            mosaic_prob=mosaic_prob,
            mixup_prob=mixup_prob,
        )
    else:
        dataset = CustomDetectionDataset(
            img_dir=cfg["img_dir"],
            label_dir=cfg["label_dir"],
            class_names=cfg["class_names"],
            img_size=cfg["img_size"],
            augment=True,
            mosaic_prob=mosaic_prob,
            mixup_prob=mixup_prob,
        )

    num_classes = dataset.num_classes
    print(f"Classes: {num_classes}  |  Samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Model ----
    anchors   = cfg.get("anchors", DEFAULT_ANCHORS)
    decoupled = cfg.get("decoupled", True)
    model = CustomDetector(num_classes=num_classes, anchors=anchors, decoupled=decoupled).to(device)
    print(f"Parameters: {model.num_parameters() / 1e6:.1f}M")

    # ---- Resume ----
    start_epoch = 0
    save_dir = Path(cfg.get("save_dir", "runs/train"))
    save_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("resume"):
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {cfg['resume']} (epoch {start_epoch})")

    # ---- Loss / Optimiser / Scheduler ----
    criterion = DetectionLoss(
        anchors=anchors,
        strides=STRIDES,
        num_classes=num_classes,
    ).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.get("lr", 0.01),
        momentum=0.937,
        weight_decay=5e-4,
        nesterov=True,
    )

    total_epochs = cfg.get("epochs", 100)
    # Pass last_epoch so the scheduler starts at the correct position when resuming
    # (avoids the "step() before optimizer.step()" warning from the fast-forward loop)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: cosine_lr_lambda(ep, total_epochs),
        last_epoch=start_epoch - 1,
    )

    # ---- Training loop ----
    best_loss = float("inf")
    img_size  = cfg["img_size"]

    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for step, (images, targets) in enumerate(loader):
            images  = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss, loss_dict = criterion(preds, targets, img_size)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()

            if step % 50 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"  [{epoch}/{total_epochs}] step {step:4d}/{len(loader)}  "
                    f"loss={loss.item():.4f}  "
                    f"box={loss_dict['box']:.4f}  "
                    f"obj={loss_dict['obj']:.4f}  "
                    f"cls={loss_dict['cls']:.4f}  "
                    f"lr={lr_now:.6f}"
                )

        scheduler.step()

        avg_loss = running_loss / len(loader)
        elapsed  = time.time() - t0
        print(f"Epoch {epoch:3d}  avg_loss={avg_loss:.4f}  ({elapsed:.1f}s)")

        # Save checkpoint
        ckpt = {
            "epoch":       epoch,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "loss":        avg_loss,
            "num_classes": num_classes,
            "anchors":     anchors,
            "img_size":    img_size,
            "class_names": getattr(dataset, "class_names", []),
        }
        torch.save(ckpt, save_dir / "last.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  >> Best checkpoint saved (loss={best_loss:.4f})")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train custom object detector")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    # CLI overrides (take precedence over config file)
    parser.add_argument("--data-type",    choices=["coco", "custom"])
    parser.add_argument("--data-root",    help="COCO dataset root")
    parser.add_argument("--split",        default=None)
    parser.add_argument("--img-dir",      help="Custom dataset image directory")
    parser.add_argument("--label-dir",    help="Custom dataset label directory")
    parser.add_argument("--class-names",  nargs="+", help="Class names for custom dataset")
    parser.add_argument("--epochs",       type=int)
    parser.add_argument("--batch-size",   type=int)
    parser.add_argument("--lr",           type=float)
    parser.add_argument("--img-size",     type=int)
    parser.add_argument("--resume",       help="Path to checkpoint to resume from")
    parser.add_argument("--device",       help="'cpu', 'cuda', 'cuda:0', etc.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    overrides = {
        "data_type":   args.data_type,
        "data_root":   args.data_root,
        "split":       args.split,
        "img_dir":     args.img_dir,
        "label_dir":   args.label_dir,
        "class_names": args.class_names,
        "epochs":      args.epochs,
        "batch_size":  args.batch_size,
        "lr":          args.lr,
        "img_size":    args.img_size,
        "resume":      args.resume,
        "device":      args.device,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    train(cfg)


if __name__ == "__main__":
    main()
