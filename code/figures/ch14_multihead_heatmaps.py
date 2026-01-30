"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 14 — Multi-head attention stacked heatmaps (clean)

Generates a single-column figure of stacked attention heatmaps by head.
No shapes diagram (kept in text for clarity).

Saves: figures/ch14_multihead_heatmaps.svg
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from code.figures._save import save_png_pdf


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


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def make_data(T: int = 8, d_model: int = 8, h: int = 4, seed: int = 1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(T, d_model))
    Wq = rng.normal(size=(d_model, d_model))
    Wk = rng.normal(size=(d_model, d_model))
    Wv = rng.normal(size=(d_model, d_model))
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv
    d_head = d_model // h
    # reshape to (h, T, d_head)
    Qh = Q.reshape(T, h, d_head).transpose(1, 0, 2)
    Kh = K.reshape(T, h, d_head).transpose(1, 0, 2)
    Vh = V.reshape(T, h, d_head).transpose(1, 0, 2)
    # scores (h, T, T)
    S = np.matmul(Qh, np.transpose(Kh, (0, 2, 1))) / np.sqrt(d_head)
    A = softmax(S, axis=-1)
    return A, (T, d_model, h, d_head)


def main(out: str = "figures/ch14_multihead_heatmaps.svg") -> None:
    _apply_style()
    A, (T, d_model, h, d_head) = make_data(T=8, d_model=8, h=4, seed=2)

    # 2x2 grid of heads (assumes h>=4; if fewer, fill available)
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4, 3.6),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    v_max = float(A.max())
    for idx in range(min(h, nrows * ncols)):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        im = ax.imshow(A[idx], vmin=0, vmax=v_max, cmap="RdBu")
        ax.set_title(f"Head {idx+1}", fontsize=8, pad=1)
        ax.set_ylabel("query i")
        if r == nrows - 1:
            ax.set_xlabel("key j")
        ax.set_xticks(np.arange(-0.5, T, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, T, 1), minor=True)
        ax.grid(
            which="minor",
            color="#dddddd",
            linestyle="-",
            linewidth=0.4,
            alpha=0.6,
        )
    # If extra subplots exist (h<4), hide unused axes
    for idx in range(h, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.savefig(out, bbox_inches="tight")
    png_path = out[:-4] + ".png" if out.endswith(".svg") else out + ".png"
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved {out} and {png_path}")
    fig.savefig(out, bbox_inches="tight")
    save_png_pdf(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
