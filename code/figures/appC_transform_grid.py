"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Appendix C — Visualize a 2x2 linear map acting on a grid and unit circle.

Output: figures/appC_transform_grid.svg
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
    out = Path("figures/appC_transform_grid.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    A = np.array([[1.2, 0.4], [0.2, 1.0]])

    # Grid lines
    xs = np.linspace(-2, 2, 9)
    ys = np.linspace(-2, 2, 9)

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for x in xs:
        line = np.stack([np.full_like(ys, x), ys], axis=1)  # vertical
        tline = (A @ line.T).T
        ax.plot(tline[:, 0], tline[:, 1], color='gray', alpha=0.25, lw=0.8)
    for y in ys:
        line = np.stack([xs, np.full_like(xs, y)], axis=1)  # horizontal
        tline = (A @ line.T).T
        ax.plot(tline[:, 0], tline[:, 1], color='gray', alpha=0.25, lw=0.8)

    # Unit circle -> ellipse
    theta = np.linspace(0, 2*np.pi, 300)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    ellipse = (A @ circle.T).T
    ax.plot(ellipse[:, 0], ellipse[:, 1], color='C2', lw=1.6, label='A·unit circle')

    ax.set_aspect('equal')
    ax.set_xlim(-3.0, 3.0); ax.set_ylim(-3.0, 3.0)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, format='svg')
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

