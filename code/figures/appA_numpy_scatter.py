"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Appendix A — Generate a scatter SVG demonstrating a broadcasted transform.

Output: figures/appA_numpy_scatter.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')


def main() -> None:
    out = Path("figures/appA_numpy_scatter.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    pts = rng.normal(size=(300, 2))
    # Broadcast a shift and scale
    shift = np.array([2.0, -1.0])
    scale = np.array([1.5, 0.5])
    pts2 = pts*scale + shift

    plt.figure(figsize=(4.2, 3.2))
    plt.scatter(pts[:,0], pts[:,1], s=15, alpha=0.5, label='original')
    plt.scatter(pts2[:,0], pts2[:,1], s=15, alpha=0.5, label='broadcasted')
    plt.axis('equal'); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

