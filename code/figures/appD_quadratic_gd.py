"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Appendix D — Gradient descent path on a convex quadratic f(x,y) = x^2 + 0.5 y^2.

Output: figures/appD_quadratic_gd.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from code.figures._save import save_png_pdf
plt.style.use('seaborn-v0_8')


def f(x, y):
    return x * x + 0.5 * y * y


def grad(x, y):
    return np.array([2 * x, y])


def main() -> None:
    out = Path("figures/appD_quadratic_gd.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Contours
    gx = np.linspace(-3.5, 0.5, 200)
    gy = np.linspace(-2.5, 2.5, 200)
    XX, YY = np.meshgrid(gx, gy)
    ZZ = f(XX, YY)

    # GD path
    xy = np.array([-3.0, 2.0])
    eta = 0.2
    pts = [xy.copy()]
    for _ in range(18):
        xy = xy - eta * grad(*xy)
        pts.append(xy.copy())
    pts = np.array(pts)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    cs = ax.contour(XX, YY, ZZ, levels=12, cmap='Greys', alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

    ax.plot(pts[:, 0], pts[:, 1], 'o-', color='C0', ms=3.5, lw=1.2, label='GD path')
    ax.scatter([0], [0], c='k', s=20, label='min')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, format='svg')
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

