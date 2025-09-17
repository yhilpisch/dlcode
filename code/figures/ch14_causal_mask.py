"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 14 — Causal mask triangular heatmap

Shows a T x T attention mask where positions j>i are blocked (upper triangle).
Saves: figures/ch14_causal_mask.svg
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


def main(T: int = 10, out: str = "figures/ch14_causal_mask.svg") -> None:
    _apply_style()
    mask = np.tril(np.ones((T, T)))  # 1 where allowed, 0 where blocked

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(mask, cmap="RdBu", vmin=0, vmax=1, origin="upper", interpolation="nearest")
    ax.set_xlabel("key index j")
    ax.set_ylabel("query index i")
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    # Thin grid lines
    ax.set_xticks(np.arange(-.5, T, 1), minor=True)
    ax.set_yticks(np.arange(-.5, T, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linestyle="-", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    png_path = out[:-4] + '.png' if out.endswith('.svg') else out + '.png'
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved {out} and {png_path}")


if __name__ == "__main__":
    main()
