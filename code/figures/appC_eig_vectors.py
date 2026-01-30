"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Appendix C — Plot eigenvector directions for a symmetric 2x2 matrix.

Output: figures/appC_eig_vectors.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from code.figures._save import save_png_pdf
plt.style.use('seaborn-v0_8')


def arrow(ax, vec, **kw):
    vx, vy = vec
    ax.arrow(
        0,
        0,
        vx,
        vy,
        head_width=0.1,
        length_includes_head=True,
        linewidth=2.4,
        **kw,
    )


def main() -> None:
    out = Path("figures/appC_eig_vectors.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    S = np.array([[2.0, 1.0], [1.0, 3.0]])
    w, Q = np.linalg.eigh(S)  # Q columns are eigenvectors

    v1 = Q[:, 0] * w[0]**0.25  # scale slightly for visibility
    v2 = Q[:, 1] * w[1]**0.25

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    arrow(ax, v1, color='C0')
    arrow(ax, v2, color='C1')
    ax.set_aspect('equal')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(
        [f"eigvec1 (λ≈{w[0]:.2f})", f"eigvec2 (λ≈{w[1]:.2f})"],
        frameon=False,
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(out, format="svg")
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
