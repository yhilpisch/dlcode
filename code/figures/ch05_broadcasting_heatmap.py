"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 5 — Broadcasting heatmap illustration (SVG).

Output: figures/ch05_broadcasting_heatmap.svg
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
    out = Path("figures/ch05_broadcasting_heatmap.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    a = np.arange(3).reshape(3, 1)
    b = np.arange(4).reshape(1, 4)
    c = a + b

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    im = ax.imshow(c, cmap='viridis')
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            ax.text(
                j,
                i,
                str(c[i, j]),
                ha="center",
                va="center",
                color="white",
                fontsize=11,
            )
    ax.set_xlabel('b shape (1,4)')
    ax.set_ylabel('a shape (3,1)')
    ax.set_title('Broadcasted sum shape (3,4)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
