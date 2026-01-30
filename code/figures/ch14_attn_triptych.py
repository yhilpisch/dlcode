"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 14 — Attention Triptych (score -> softmax -> mix)

Generates a 1x3 panel with restrained styling:
- Scores (dot products) for one query against all keys
- Softmax weights (sum=1)
- Output value vector components (weighted sum)

Saves: figures/ch14_attn_triptych.svg
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from code.figures._save import save_png_pdf


def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = x / max(temp, 1e-8)
    x = x - x.max()
    ex = np.exp(x)
    return ex / ex.sum()


def _apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def main(
    seed: int = 0,
    T: int = 6,
    d: int = 4,
    i: int = 2,
    temp: float = 1.0,
    out: str = "figures/ch14_attn_triptych.svg",
) -> None:
    _apply_style()
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(T, d))
    K = rng.normal(size=(T, d))
    V = rng.normal(size=(T, d))

    s = (Q[i] @ K.T) / np.sqrt(d)
    a = softmax(s, temp=temp)
    o = a @ V  # (d,)

    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))

    # Panel 1: scores
    axs[0].bar(np.arange(T), s, color="#377eb8")  # blue
    axs[0].set_title("Scores", fontsize=9)
    axs[0].set_xticks(range(T))
    axs[0].set_xlabel("key index j")
    axs[0].axhline(0, color="#888", lw=0.8)

    # Panel 2: softmax weights
    axs[1].bar(np.arange(T), a, color="#e41a1c")  # red
    axs[1].set_title("Softmax weights", fontsize=9)
    axs[1].set_xticks(range(T))
    axs[1].set_xlabel("key index j")
    axs[1].set_ylim(0, max(0.25, a.max() * 1.15))

    # Panel 3: output vector components
    axs[2].bar(np.arange(d), o, color="#08519c")  # dark blue
    axs[2].set_title("Output components", fontsize=9)
    axs[2].set_xticks(range(d))
    axs[2].set_xlabel("dimension")
    axs[2].axhline(0, color="#888", lw=0.8)

    for ax in axs:
        ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)

    fig.tight_layout()
    # Save both SVG (vector) and PNG at 300 DPI
    fig.savefig(out, bbox_inches="tight")
    if out.endswith(".svg"):
        png_path = out[:-4] + ".png"
    else:
        png_path = out + ".png"
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    save_png_pdf(out)
    print(f"Saved {out} and {png_path}")


if __name__ == "__main__":
    main()
