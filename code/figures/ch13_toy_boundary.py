"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 13 — Mean-embedding vs BoW boundary (SVG, toy).

Output: figures/ch13_toy_boundary.svg
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from code.figures._save import save_png_pdf
plt.style.use('seaborn-v0_8')

def main() -> None:
    out = Path("figures/ch13_toy_boundary.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(1)
    # 2D toy: x ~ sentiment axis, y ~ noise
    # Slightly tilted clusters to justify a slanted boundary
    x_pos = rng.randn(40, 2) * [0.3, 0.4] + [1.2, 0.3]
    x_neg = rng.randn(40, 2) * [0.3, 0.4] + [-1.2, -0.3]
    X = np.vstack([x_pos, x_neg])
    y = np.array([1] * len(x_pos) + [0] * len(x_neg))
    # Compute an LDA-like linear boundary from class means and pooled covariance
    xx = np.linspace(-2, 2, 200)
    mu_pos = x_pos.mean(axis=0)
    mu_neg = x_neg.mean(axis=0)
    Xc_pos = x_pos - mu_pos
    Xc_neg = x_neg - mu_neg
    S = (Xc_pos.T @ Xc_pos + Xc_neg.T @ Xc_neg) / (
        len(x_pos) + len(x_neg) - 2
    )
    # Regularize slightly for numerical stability
    S += 1e-6 * np.eye(2)
    w = np.linalg.solve(S, (mu_pos - mu_neg))  # direction
    m = 0.5 * (mu_pos + mu_neg)  # midpoint
    # Decision boundary: (x - m)·w = 0  =>  w_x x + w_y y + b = 0
    # with b = - w·m
    b = - float(w @ m)
    fig, ax = plt.subplots(figsize=(5.6,4.0))
    ax.scatter(
        x_pos[:, 0], x_pos[:, 1], c="tab:green", label="positive", alpha=0.8
    )
    ax.scatter(
        x_neg[:, 0], x_neg[:, 1], c="tab:red", label="negative", alpha=0.8
    )
    ax.axvline(0.0, linestyle="--", color="k", label="BoW baseline")
    # Mean-embedding (LDA-like) boundary: y = (-w_x x - b)/w_y
    mean = (-(w[0] * xx) - b) / w[1]
    ax.plot(xx, mean, "-", color="k", label="Mean-embedding")
    ax.legend(frameon=False)
    ax.set_title("Toy decision boundaries")
    ax.set_xlabel("sentiment axis (toy)")
    ax.set_ylabel("noise")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    print(f"Wrote {out}")
    save_png_pdf(out)

if __name__ == "__main__":
    main()
