"""
Tests for model components: backbone, neck, head, full detector.

Run:
    pytest tests/ -v
"""
import torch
import pytest


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

def test_backbone_output_shapes():
    from model.backbone import Backbone
    m = Backbone()
    x = torch.zeros(2, 3, 640, 640)
    c3, c4, c5 = m(x)
    assert c3.shape == (2, 256, 80, 80), f"C3 shape mismatch: {c3.shape}"
    assert c4.shape == (2, 512, 40, 40), f"C4 shape mismatch: {c4.shape}"
    assert c5.shape == (2, 1024, 20, 20), f"C5 shape mismatch: {c5.shape}"


def test_backbone_smaller_input():
    from model.backbone import Backbone
    m = Backbone()
    x = torch.zeros(1, 3, 320, 320)
    c3, c4, c5 = m(x)
    assert c3.shape == (1, 256, 40, 40)
    assert c4.shape == (1, 512, 20, 20)
    assert c5.shape == (1, 1024, 10, 10)


# ---------------------------------------------------------------------------
# Neck (FPN)
# ---------------------------------------------------------------------------

def test_fpn_output_shapes():
    from model.backbone import Backbone
    from model.neck import FPN
    backbone = Backbone()
    fpn = FPN()
    x = torch.zeros(2, 3, 640, 640)
    c3, c4, c5 = backbone(x)
    p3, p4, p5 = fpn(c3, c4, c5)
    assert p3.shape == (2, 256, 80, 80)
    assert p4.shape == (2, 256, 40, 40)
    assert p5.shape == (2, 256, 20, 20)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decoupled", [True, False])
def test_head_output_shape(decoupled):
    from model.head import MultiScaleHead
    head = MultiScaleHead(in_ch=256, num_anchors=3, num_classes=80, decoupled=decoupled)
    p3 = torch.zeros(2, 256, 80, 80)
    p4 = torch.zeros(2, 256, 40, 40)
    p5 = torch.zeros(2, 256, 20, 20)
    o3, o4, o5 = head(p3, p4, p5)
    na, nc = 3, 80
    assert o3.shape == (2, na * (5 + nc), 80, 80)
    assert o4.shape == (2, na * (5 + nc), 40, 40)
    assert o5.shape == (2, na * (5 + nc), 20, 20)


# ---------------------------------------------------------------------------
# Full detector — forward pass
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decoupled", [True, False])
def test_detector_forward(decoupled):
    from model.detector import CustomDetector
    model = CustomDetector(num_classes=10, decoupled=decoupled)
    x = torch.zeros(1, 3, 640, 640)
    outs = model(x)
    assert len(outs) == 3
    na, nc = 3, 10
    assert outs[0].shape == (1, na * (5 + nc), 80, 80)
    assert outs[1].shape == (1, na * (5 + nc), 40, 40)
    assert outs[2].shape == (1, na * (5 + nc), 20, 20)


def test_detector_predict_returns_list():
    from model.detector import CustomDetector
    model = CustomDetector(num_classes=10)
    x = torch.rand(2, 3, 640, 640)
    results = model.predict(x, conf_thresh=0.01)
    assert len(results) == 2                # one entry per image
    for r in results:
        assert "boxes"   in r
        assert "scores"  in r
        assert "classes" in r
        assert r["boxes"].ndim == 2
        assert r["boxes"].shape[1] == 4


def test_detector_num_parameters():
    from model.detector import CustomDetector
    model = CustomDetector(num_classes=80)
    n = model.num_parameters()
    # Rough sanity check: should be in the tens of millions
    assert 5_000_000 < n < 200_000_000, f"Unexpected param count: {n}"
