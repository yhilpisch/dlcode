"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Generate SVG figure for Chapter 1: minimal linear regression fit.

Output:
  figures/ch01_linear_regression.svg
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
    out = Path("figures/ch01_linear_regression.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0.0, 1.0, 2.1, 2.9], dtype=float)

    # Simple closed-form fit for a line via numpy (to avoid sklearn dependency here)
    # y = a*x + b
    x = X.ravel()
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    xx = np.linspace(x.min(), x.max(), 100)
    yy = a * xx + b

    plt.figure(figsize=(4.2, 3.2))
    plt.scatter(x, y, label="data")
    plt.plot(xx, yy, "r-", label="fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out, format="svg")
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
