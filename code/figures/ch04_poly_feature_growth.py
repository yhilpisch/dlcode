"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 4 — Polynomial feature count growth (SVG).

Output: figures/ch04_poly_feature_growth.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from code.figures._save import save_png_pdf
plt.style.use('seaborn-v0_8')


def poly_feature_count(d: int, degree: int) -> int:
    # Number of polynomial features with interaction_only=False, include_bias=False
    # equals C(d + degree, degree) - 1 (excluding bias), summed across degrees
    # 1..degree. Efficiently compute using combinations with replacement formula.
    total = 0
    for k in range(1, degree + 1):
        total += math.comb(d + k - 1, k)
    return total


def main() -> None:
    out = Path("figures/ch04_poly_feature_growth.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    d = 20
    degrees = np.arange(1, 8)
    counts = [poly_feature_count(d, int(K)) for K in degrees]

    plt.figure(figsize=(5.0, 3.2))
    plt.plot(degrees, counts, marker='o')
    plt.yscale('log')
    plt.xlabel("polynomial degree K")
    plt.ylabel("# features (log scale)")
    plt.title(f'Polynomial features explode (d={d})')
    plt.tight_layout()
    plt.savefig(out, format="svg")
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
