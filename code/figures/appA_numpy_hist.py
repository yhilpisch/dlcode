"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Appendix A — Generate a histogram SVG of standard normal samples.

Output: figures/appA_numpy_hist.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')


def main() -> None:
    out = Path("figures/appA_numpy_hist.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    x = rng.normal(size=10_000)

    plt.figure(figsize=(4.2, 3.2))
    plt.hist(x, bins=40, density=True, alpha=0.7, color='C0')
    plt.xlabel('value'); plt.ylabel('density')
    plt.tight_layout(); plt.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

