"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 3 — Two-moons dataset scatter (SVG).

Output: figures/ch03_moons_scatter.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.datasets import make_moons
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')


def main() -> None:
    out = Path("figures/ch03_moons_scatter.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for cls, marker, label in [(0, "o", "class 0"), (1, "^", "class 1")]:
        idx = y == cls
        ax.scatter(X[idx, 0], X[idx, 1], marker=marker, s=20, alpha=0.9, label=label)
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, format="svg")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

