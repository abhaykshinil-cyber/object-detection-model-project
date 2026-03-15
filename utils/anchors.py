"""
K-means anchor generation from a label directory.

Clusters GT box shapes (w, h) to find the best anchor sizes for your
specific dataset.  Run this before training on custom data.

Usage:
    python utils/anchors.py --label-dir data/labels/train --n 9 --img-size 640
"""
import argparse
from pathlib import Path

import numpy as np


def wh_iou(wh1: np.ndarray, wh2: np.ndarray) -> np.ndarray:
    """
    IoU between two sets of (w, h) pairs, treating both as centred at origin.

    Args:
        wh1: (N, 2)
        wh2: (K, 2)

    Returns:
        (N, K) IoU matrix
    """
    w1, h1 = wh1[:, 0:1], wh1[:, 1:2]
    w2, h2 = wh2[:, 0:1].T, wh2[:, 1:2].T
    inter = np.minimum(w1, w2) * np.minimum(h1, h2)
    union = w1 * h1 + w2 * h2 - inter
    return inter / (union + 1e-8)


def kmeans_anchors(
    label_dir: str,
    n: int = 9,
    img_size: int = 640,
    n_iter: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """
    Run k-means clustering on GT box shapes to compute optimal anchors.

    Args:
        label_dir: directory of YOLO-format .txt label files
        n:         number of anchors to compute (typically 9 for 3 scales × 3)
        img_size:  training image size (anchors are scaled to this resolution)
        n_iter:    maximum k-means iterations
        seed:      random seed

    Returns:
        (n, 2) int array of (w, h) anchor sizes in pixels at img_size
    """
    wh_list = []
    for txt in sorted(Path(label_dir).glob("*.txt")):
        rows = np.loadtxt(str(txt), dtype=np.float32)
        if rows.ndim == 1:
            rows = rows[None]
        if rows.size > 0:
            wh_list.append(rows[:, 3:5])   # normalised w, h

    if not wh_list:
        raise ValueError(f"No label files found in {label_dir}")

    wh = np.concatenate(wh_list) * img_size   # absolute pixels

    rng = np.random.default_rng(seed)
    centroids = wh[rng.choice(len(wh), n, replace=False)].copy()

    for iteration in range(n_iter):
        dist = 1.0 - wh_iou(wh, centroids)    # (N, k)  — distance = 1 - IoU
        assignments = dist.argmin(axis=1)

        new_centroids = np.array([
            wh[assignments == k].mean(axis=0) if (assignments == k).any() else centroids[k]
            for k in range(n)
        ])

        if np.allclose(new_centroids, centroids, atol=0.1):
            print(f"  Converged after {iteration + 1} iterations.")
            break
        centroids = new_centroids

    # Sort anchors by area (smallest first)
    order = np.argsort(centroids[:, 0] * centroids[:, 1])
    centroids = centroids[order]

    avg_iou = wh_iou(wh, centroids).max(axis=1).mean()
    print(f"  Average best anchor IoU: {avg_iou:.3f}")

    return centroids.round().astype(int)


def format_anchors(anchors: np.ndarray) -> str:
    """Format anchors array as YAML-ready string grouped into 3 scales."""
    groups = [anchors[:3].tolist(), anchors[3:6].tolist(), anchors[6:].tolist()]
    lines = ["anchors:"]
    for g in groups:
        pairs = ", ".join(f"[{int(w)}, {int(h)}]" for w, h in g)
        lines.append(f"  - [{pairs}]")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute k-means anchors from label files")
    parser.add_argument("--label-dir", required=True, help="Directory of .txt label files")
    parser.add_argument("--n", type=int, default=9, help="Number of anchors (default: 9)")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Computing {args.n} anchors from {args.label_dir} …")
    anchors = kmeans_anchors(args.label_dir, args.n, args.img_size, seed=args.seed)

    print("\nAnchors (w × h in pixels):")
    for i, (w, h) in enumerate(anchors):
        print(f"  Anchor {i + 1:2d}: ({w:4d}, {h:4d})")

    print(f"\n{format_anchors(anchors)}")
