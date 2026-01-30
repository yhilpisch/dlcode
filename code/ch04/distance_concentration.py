"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 4 — Estimate distance contrast as dimensionality grows.

Run:
  python code/ch04/distance_concentration.py
"""
from __future__ import annotations

import numpy as np


def relative_contrast(
    dim: int,
    rng: np.random.Generator,
    n_points: int = 2000,
    sample: int = 120,
) -> float:
    """Return the relative contrast (max - min) / min of pairwise distances.

    The contrast is computed from pairwise distances for sampled anchors.
    """
    X = rng.random((n_points, dim))
    anchors = rng.choice(n_points, size=min(sample, n_points - 1), replace=False)
    mins, maxs = [], []

    for idx in anchors:
        dists = np.linalg.norm(X - X[idx], axis=1)
        dists = dists[dists > 0]  # drop self-distance
        mins.append(dists.min())
        maxs.append(dists.max())

    return (np.mean(maxs) - np.mean(mins)) / np.mean(mins)


def main() -> None:
    rng = np.random.default_rng(0)
    dims = np.array([2, 5, 10, 20, 50, 100])
    contrasts = [relative_contrast(int(d), rng) for d in dims]

    print("dimension,relative_contrast")
    for d, rc in zip(dims, contrasts):
        print(f"{int(d)},{rc:.3f}")


if __name__ == "__main__":
    main()
