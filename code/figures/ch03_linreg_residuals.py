"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 3 — Residuals vs x for linear regression (SVG).

Output: figures/ch03_linreg_residuals.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')


def main() -> None:
    out = Path("figures/ch03_linreg_residuals.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    X = np.linspace(0, 5, 30).reshape(-1, 1)
    y = 1.2 * X.ravel() + 0.5 + rng.normal(0, 0.35, size=X.shape[0])

    # Fit via least squares (closed form for line)
    x = X.ravel()
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = a * x + b
    resid = y - yhat

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.scatter(x, resid, s=22, alpha=0.9)
    ax.axhline(0, color='k', lw=1.2, alpha=0.7)
    ax.set_xlabel('x'); ax.set_ylabel('residual (y - yhat)')
    fig.tight_layout(); fig.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

