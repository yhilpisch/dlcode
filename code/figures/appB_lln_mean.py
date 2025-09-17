"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Appendix B — Generate running mean (LLN) SVG for N(0,1).

Output: figures/appB_lln_mean.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')


def main() -> None:
    out = Path("figures/appB_lln_mean.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    run = np.cumsum(x)/np.arange(1, x.size+1)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(run, label='running mean')
    ax.axhline(0.0, color='k', lw=1, alpha=0.6)
    ax.set_xlabel('n'); ax.set_ylabel('mean of first n samples')
    ax.set_title('LLN: running mean of N(0,1)')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

