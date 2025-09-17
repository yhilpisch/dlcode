"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Appendix C — Plot two 2D vectors and a linear combination.

Output: figures/appC_vectors_combo.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')


def arrow(ax, origin, vec, **kw):
    ox, oy = origin
    vx, vy = vec
    ax.arrow(ox, oy, vx, vy, head_width=0.1, length_includes_head=True, linewidth=2.4, **kw)


def main() -> None:
    out = Path("figures/appC_vectors_combo.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    u = np.array([2.0, 1.0])
    v = np.array([1.0, 3.0])
    a, b = 0.5, 1.2
    w = a*u + b*v

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    arrow(ax, (0, 0), u, color='C0')
    arrow(ax, (0, 0), v, color='C1')
    arrow(ax, (0, 0), w, color='C2')
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 5.0); ax.set_ylim(-0.5, 5.0)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(['u', 'v', f'{a}u+{b}v'], frameon=False, loc='upper right')
    fig.tight_layout(); fig.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
