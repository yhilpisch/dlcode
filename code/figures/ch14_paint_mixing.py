"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 14 — Paint mixing demo for attention values

Values are mapped to RGB colors; attention weights mix them into an output color.
Saves: figures/ch14_paint_mixing.svg
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
        }
    )


def main(
    seed: int = 3,
    T: int = 5,
    d: int = 6,
    i: int = 1,
    temp: float = 1.0,
    out: str = "figures/ch14_paint_mixing.svg",
) -> None:
    _apply_style()
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(T, d))
    K = rng.normal(size=(T, d))
    V = rng.normal(size=(T, d))

    # Row i attention
    s = (Q[i] @ K.T) / np.sqrt(d)
    a = softmax(s, temp=temp)

    # Map tokens to blue→red colors along a gradient for clarity
    cmap = plt.get_cmap("RdBu")
    colors = np.array([cmap(t) for t in np.linspace(0.1, 0.9, T)])[:, :3]
    mix = a @ colors  # (3,)

    fig = plt.figure(figsize=(6.2, 2.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.6, 1.2], wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(np.arange(T), a, color="#e41a1c")  # red
    ax1.set_title("Weights", fontsize=9)
    ax1.set_xticks(range(T))
    ax1.set_xlabel("token j")
    ax1.set_ylim(0, max(0.25, a.max() * 1.2))
    ax1.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)

    ax2 = fig.add_subplot(gs[0, 1])
    for j in range(T):
        ax2.add_patch(
            plt.Rectangle(
                (0.1, j + 0.15), 0.8, 0.7, color=colors[j], ec="#333"
            )
        )
        ax2.text(0.92, j + 0.5, f"j={j}", va="center", ha="left", fontsize=8)
    ax2.set_xlim(0, 1.2)
    ax2.set_ylim(0, T)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_title("Token colors", fontsize=9)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.add_patch(
        plt.Rectangle((0.1, 0.35), 0.8, 0.7, color=mix, ec="#333", lw=1.5)
    )
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1.5)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title("Mixed color", fontsize=9)
    for spine in ax3.spines.values():
        spine.set_visible(False)

    # constrained_layout avoids tight_layout warnings with mixed artists
    fig.savefig(out, bbox_inches="tight")
    png_path = out[:-4] + ".png" if out.endswith(".svg") else out + ".png"
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    save_png_pdf(out)
    print(f"Saved {out} and {png_path}")


if __name__ == "__main__":
    main()
