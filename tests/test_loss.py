"""
Tests for loss functions: IoU variants and DetectionLoss.
"""
import torch
import pytest


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def test_iou_perfect_overlap():
    from loss.iou import bbox_iou
    boxes = torch.tensor([[0., 0., 10., 10.]])
    iou = bbox_iou(boxes, boxes, mode="iou")
    assert torch.allclose(iou, torch.ones(1)), f"Expected 1.0, got {iou}"


def test_iou_no_overlap():
    from loss.iou import bbox_iou
    b1 = torch.tensor([[0., 0., 5., 5.]])
    b2 = torch.tensor([[10., 10., 20., 20.]])
    iou = bbox_iou(b1, b2, mode="iou")
    assert iou.item() == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("mode", ["iou", "giou", "diou", "ciou"])
def test_iou_modes_run(mode):
    from loss.iou import bbox_iou
    b1 = torch.rand(8, 4)
    b2 = torch.rand(8, 4)
    # Ensure x2 > x1, y2 > y1
    b1[:, 2:] = b1[:, :2] + b1[:, 2:].abs() + 0.1
    b2[:, 2:] = b2[:, :2] + b2[:, 2:].abs() + 0.1
    result = bbox_iou(b1, b2, mode=mode)
    assert result.shape == (8,)
    assert not result.isnan().any()


def test_ciou_loss_decreases():
    """CIoU loss should decrease when predicted box is moved toward GT."""
    from loss.iou import bbox_iou
    gt = torch.tensor([[50., 50., 150., 150.]])
    # Start far away
    pred_far  = torch.tensor([[200., 200., 300., 300.]])
    # Start close
    pred_near = torch.tensor([[60., 60., 140., 140.]])
    loss_far  = 1 - bbox_iou(pred_far,  gt, mode="ciou")
    loss_near = 1 - bbox_iou(pred_near, gt, mode="ciou")
    assert loss_near.item() < loss_far.item()


# ---------------------------------------------------------------------------
# DetectionLoss
# ---------------------------------------------------------------------------

def _make_targets(B=2, n_gt=4, num_classes=10):
    """Create random normalised targets: (n_gt*B, 6) [batch_idx, cls, cx, cy, w, h]."""
    targets = []
    for b in range(B):
        for _ in range(n_gt):
            cx = torch.rand(1).item()
            cy = torch.rand(1).item()
            w  = torch.rand(1).item() * 0.3 + 0.05
            h  = torch.rand(1).item() * 0.3 + 0.05
            cls = torch.randint(0, num_classes, (1,)).item()
            targets.append([b, cls, cx, cy, w, h])
    return torch.tensor(targets, dtype=torch.float32)


def test_detection_loss_forward():
    from model.detector import CustomDetector
    from loss.detection_loss import DetectionLoss
    from model.detector import DEFAULT_ANCHORS, STRIDES

    nc = 10
    model = CustomDetector(num_classes=nc)
    x = torch.rand(2, 3, 640, 640)
    preds = model(x)

    criterion = DetectionLoss(anchors=DEFAULT_ANCHORS, strides=STRIDES, num_classes=nc)
    targets = _make_targets(B=2, n_gt=3, num_classes=nc)
    loss, components = criterion(preds, targets, img_size=640)

    assert loss.item() > 0
    assert not loss.isnan()
    assert "box" in components
    assert "obj" in components
    assert "cls" in components


def test_detection_loss_empty_targets():
    """Loss should still compute (obj loss only) when there are no GT boxes."""
    from model.detector import CustomDetector
    from loss.detection_loss import DetectionLoss
    from model.detector import DEFAULT_ANCHORS, STRIDES

    nc = 5
    model = CustomDetector(num_classes=nc)
    x = torch.rand(1, 3, 640, 640)
    preds = model(x)

    criterion = DetectionLoss(anchors=DEFAULT_ANCHORS, strides=STRIDES, num_classes=nc)
    empty = torch.zeros((0, 6))
    loss, _ = criterion(preds, empty, img_size=640)
    assert loss.item() >= 0
    assert not loss.isnan()


def test_detection_loss_backward():
    """Gradients should flow through all parameters."""
    from model.detector import CustomDetector
    from loss.detection_loss import DetectionLoss
    from model.detector import DEFAULT_ANCHORS, STRIDES

    nc = 5
    model = CustomDetector(num_classes=nc)
    x = torch.rand(1, 3, 320, 320)
    preds = model(x)
    targets = _make_targets(B=1, n_gt=2, num_classes=nc)

    criterion = DetectionLoss(anchors=DEFAULT_ANCHORS, strides=STRIDES, num_classes=nc)
    loss, _ = criterion(preds, targets, img_size=320)
    loss.backward()

    grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert grad_count > 0, "No gradients flowed through the model"
