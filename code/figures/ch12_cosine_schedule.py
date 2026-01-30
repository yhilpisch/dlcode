"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 12 — Cosine vs constant learning-rate schedule (SVG/PNG).

Outputs:
- figures/ch12_cosine_schedule.svg
- figures/ch12_cosine_schedule.png
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from code.figures._save import save_png_pdf

plt.style.use("seaborn-v0_8")


def main() -> None:
    out = Path("figures/ch12_cosine_schedule.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(0, 100)
    base_lr = 5e-3
    constant = np.full_like(epochs, base_lr, dtype=float)
    cosine = 0.5 * base_lr * (1 + np.cos(np.pi * epochs / epochs.max()))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(epochs, constant, label="Constant LR", alpha=0.9)
    ax.plot(epochs, cosine, label="Cosine schedule", alpha=0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title("Cosine vs constant learning rate")
    fig.tight_layout()

    fig.savefig(out, format="svg")
    png_out = out.with_suffix(".png")
    fig.savefig(png_out, dpi=200)
    save_png_pdf(out)
    print(f"Wrote {out} and {png_out}")


if __name__ == "__main__":
    main()
