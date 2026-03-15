"""
Tests for utility functions: NMS, mAP, augmentations, anchors.
"""
import numpy as np
import torch
import pytest


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def test_nms_suppresses_duplicates():
    from utils.nms import nms
    # Two nearly identical boxes — only the higher-score one should survive
    boxes  = torch.tensor([[10., 10., 50., 50.], [11., 11., 51., 51.]])
    scores = torch.tensor([0.9, 0.7])
    kept = nms(boxes, scores, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0].item() == 0   # higher score kept


def test_nms_keeps_non_overlapping():
    from utils.nms import nms
    boxes  = torch.tensor([[0., 0., 10., 10.], [100., 100., 200., 200.]])
    scores = torch.tensor([0.9, 0.8])
    kept = nms(boxes, scores, iou_threshold=0.5)
    assert len(kept) == 2


def test_nms_empty():
    from utils.nms import nms
    kept = nms(torch.zeros((0, 4)), torch.zeros(0))
    assert len(kept) == 0


def test_multiclass_nms():
    from utils.nms import multiclass_nms
    boxes   = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.],  # cls 0 — duplicate
                             [50., 50., 60., 60.]])                    # cls 1 — unique
    scores  = torch.tensor([0.9, 0.8, 0.7])
    classes = torch.tensor([0, 0, 1])
    b, s, c = multiclass_nms(boxes, scores, classes, iou_threshold=0.5)
    assert len(b) == 2   # 1 from cls0 + 1 from cls1


# ---------------------------------------------------------------------------
# mAP
# ---------------------------------------------------------------------------

def test_map_perfect_predictions():
    from utils.metrics import compute_map
    boxes = torch.tensor([[0., 0., 10., 10.]])
    preds = [{"boxes": boxes, "scores": torch.tensor([0.95]), "classes": torch.tensor([0])}]
    gts   = [{"boxes": boxes, "classes": torch.tensor([0])}]
    result = compute_map(preds, gts, num_classes=1, iou_threshold=0.5)
    assert result["mAP"] == pytest.approx(1.0, abs=0.01)


def test_map_no_predictions():
    from utils.metrics import compute_map
    gt_boxes = torch.tensor([[0., 0., 10., 10.]])
    preds = [{"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "classes": torch.zeros(0, dtype=torch.long)}]
    gts   = [{"boxes": gt_boxes, "classes": torch.tensor([0])}]
    result = compute_map(preds, gts, num_classes=1)
    assert result["mAP"] == pytest.approx(0.0, abs=0.01)


def test_map_wrong_class():
    from utils.metrics import compute_map
    boxes = torch.tensor([[0., 0., 10., 10.]])
    preds = [{"boxes": boxes, "scores": torch.tensor([0.9]), "classes": torch.tensor([1])}]
    gts   = [{"boxes": boxes, "classes": torch.tensor([0])}]
    result = compute_map(preds, gts, num_classes=2, iou_threshold=0.5)
    assert result["AP_cls0"] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def _rand_image(h=480, w=640):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _rand_targets(n=4):
    t = np.zeros((n, 5), dtype=np.float32)
    t[:, 0] = np.random.randint(0, 3, n)
    t[:, 1:3] = np.random.uniform(0.1, 0.9, (n, 2))  # cx, cy
    t[:, 3:5] = np.random.uniform(0.05, 0.2, (n, 2))  # w, h
    return t


def test_letterbox_output_shape():
    from data.augment import letterbox
    img = _rand_image(480, 640)
    out = letterbox(img, 640)
    assert out.shape == (640, 640, 3)


def test_letterbox_square_input():
    from data.augment import letterbox
    img = _rand_image(320, 320)
    out = letterbox(img, 640)
    assert out.shape == (640, 640, 3)


def test_augmenter_output_shape():
    from data.augment import Augmenter
    aug = Augmenter(img_size=640, mosaic_prob=0.0)   # no mosaic (no buffer)
    img = _rand_image()
    tgt = _rand_targets()
    out_img, out_tgt = aug(img, tgt)
    assert out_img.shape == (640, 640, 3)
    assert out_tgt.ndim == 2
    if out_tgt.shape[0] > 0:
        assert out_tgt.shape[1] == 5


def test_augmenter_normalised_targets():
    from data.augment import Augmenter
    aug = Augmenter(img_size=640, mosaic_prob=0.0)
    img = _rand_image()
    tgt = _rand_targets(10)
    _, out_tgt = aug(img, tgt)
    if out_tgt.shape[0] > 0:
        assert out_tgt[:, 1:].min() >= 0.0
        assert out_tgt[:, 1:].max() <= 1.0


def test_mosaic_output_shape():
    from data.augment import mosaic
    samples = [(_rand_image(), _rand_targets(3)) for _ in range(4)]
    img, tgt = mosaic(samples, img_size=640)
    assert img.shape == (640, 640, 3)
    assert tgt.ndim == 2
    if tgt.shape[0] > 0:
        assert tgt.shape[1] == 5


def test_mixup_output_shape():
    from data.augment import mixup
    img1, tgt1 = _rand_image(640, 640), _rand_targets(3)
    img2, tgt2 = _rand_image(640, 640), _rand_targets(2)
    out_img, out_tgt = mixup(img1, tgt1, img2, tgt2)
    assert out_img.shape == (640, 640, 3)
    assert len(out_tgt) == len(tgt1) + len(tgt2)


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

def test_kmeans_anchors(tmp_path):
    from utils.anchors import kmeans_anchors
    # Create fake label files
    for i in range(20):
        rows = np.random.uniform(0.05, 0.5, (5, 5))
        rows[:, 0] = 0   # class index
        np.savetxt(str(tmp_path / f"{i}.txt"), rows)

    anchors = kmeans_anchors(str(tmp_path), n=9, img_size=640, n_iter=50)
    assert anchors.shape == (9, 2)
    assert (anchors > 0).all()
    # Sorted by area (smallest first)
    areas = anchors[:, 0] * anchors[:, 1]
    assert (areas[1:] >= areas[:-1]).all()
