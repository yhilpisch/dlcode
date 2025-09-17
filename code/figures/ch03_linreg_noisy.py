"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 3 — Linear regression on noisy data (SVG).

Output: figures/ch03_linreg_noisy.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')


def main() -> None:
    out = Path("figures/ch03_linreg_noisy.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    X = np.linspace(0, 5, 30).reshape(-1, 1)
    y = 1.2 * X.ravel() + 0.5 + rng.normal(0, 0.35, size=X.shape[0])

    # Closed-form via least squares for the line y = a x + b
    x = X.ravel()
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    xx = np.linspace(x.min(), x.max(), 200)
    yy = a * xx + b

    plt.figure(figsize=(4.8, 3.4))
    plt.scatter(x, y, s=20, alpha=0.8, label='data')
    plt.plot(xx, yy, 'r-', lw=2, label='fit')
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

