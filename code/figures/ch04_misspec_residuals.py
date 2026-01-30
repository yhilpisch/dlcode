"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 4 — Residual pattern under model mis-specification (SVG).

Output: figures/ch04_misspec_residuals.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
from numpy.linalg import lstsq
from code.figures._save import save_png_pdf


def main() -> None:
    out = Path("figures/ch04_misspec_residuals.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    x = np.linspace(-3, 3, 80)
    y_true = 0.6 * x**2 - 0.5 * x + 0.3
    y = y_true + rng.normal(0, 0.5, size=x.shape)

    A = np.vstack([x, np.ones_like(x)]).T
    a, b = lstsq(A, y, rcond=None)[0]
    y_lin = a * x + b
    resid = y - y_lin

    plt.figure(figsize=(5.2, 3.4))
    plt.scatter(x, resid, s=18, alpha=0.9)
    plt.axhline(0, color='k', lw=1)
    plt.xlabel('x'); plt.ylabel('residual (y - y_linear)')
    plt.title('Curved residual pattern indicates mis-specification')
    plt.tight_layout(); plt.savefig(out, format='svg')
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

