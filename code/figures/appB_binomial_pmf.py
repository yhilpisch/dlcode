"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Appendix B — Generate Binomial(n=10,p=0.3) PMF SVG.

Output: figures/appB_binomial_pmf.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from math import comb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from code.figures._save import save_png_pdf
plt.style.use('seaborn-v0_8')


def main() -> None:
    out = Path("figures/appB_binomial_pmf.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    n, p = 10, 0.3
    ks = np.arange(0, n+1)
    pmf = np.array([comb(n, int(k))*(p**k)*((1-p)**(n-k)) for k in ks])

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.bar(ks, pmf, color='C0', alpha=0.8)
    ax.set_xlabel('k (successes)'); ax.set_ylabel('P(X=k)')
    ax.set_title(f'Binomial(n={n}, p={p}) PMF')
    fig.tight_layout(); fig.savefig(out, format='svg')
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

