"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 3 — Histogram of residuals for linear regression (SVG).

Output: figures/ch03_linreg_resid_hist.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')


def main() -> None:
    out = Path("figures/ch03_linreg_resid_hist.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    X = np.linspace(0, 5, 30).reshape(-1, 1)
    y = 1.2 * X.ravel() + 0.5 + rng.normal(0, 0.35, size=X.shape[0])

    x = X.ravel()
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - (a * x + b)

    plt.figure(figsize=(4.6, 3.2))
    plt.hist(resid, bins=12, alpha=0.8, color='C0', edgecolor='white')
    plt.axvline(0, color='k', lw=1, alpha=0.7)
    plt.xlabel('residual'); plt.ylabel('count')
    plt.tight_layout(); plt.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

