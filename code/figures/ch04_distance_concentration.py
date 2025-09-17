"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 4 — Distance concentration with increasing dimension (SVG).

Output: figures/ch04_distance_concentration.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')


def main() -> None:
    out = Path("figures/ch04_distance_concentration.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    dims = np.array([2, 5, 10, 20, 50, 100])
    n_points = 500
    rel_contrast = []  # mean(max) / mean(min) or (max-min)/min
    for d in dims:
        X = rng.random((n_points, int(d)))
        # sample a subset to estimate distances efficiently
        idx = rng.choice(n_points, size=60, replace=False)
        mins, maxs = [], []
        for i in idx:
            diffs = X - X[i]
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            dists = dists[dists > 0]
            mins.append(dists.min())
            maxs.append(dists.max())
        mins = np.array(mins); maxs = np.array(maxs)
        rel_contrast.append((maxs.mean() - mins.mean()) / mins.mean())

    rel_contrast = np.array(rel_contrast)
    plt.figure(figsize=(5.0, 3.2))
    plt.plot(dims, rel_contrast, marker='o')
    plt.xlabel('dimension d'); plt.ylabel('relative contrast (avg)')
    plt.title('Distance contrast shrinks as dimension grows')
    plt.tight_layout(); plt.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

