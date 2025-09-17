"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Appendix A — Generate a sine/cosine SVG line plot.

Output: figures/appA_numpy_line.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')


def main() -> None:
    out = Path("figures/appA_numpy_line.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0, 2*np.pi, 400)
    s = np.sin(x)
    c = np.cos(x)

    plt.figure(figsize=(4.2, 3.2))
    plt.plot(x, s, label='sin(x)')
    plt.plot(x, c, label='cos(x)')
    plt.xlabel('x'); plt.ylabel('value'); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

