"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 12 — Cosine vs constant learning-rate schedule (SVG/PNG).

Outputs:
- figures/ch12_cosine_schedule.svg
- figures/ch12_cosine_schedule.png
"""
from __future__ import annotations  # postponed evaluations for older interpreters
from pathlib import Path  # filesystem utilities for output paths

import numpy as np  # numerical helpers for schedule arrays
import matplotlib  # base Matplotlib import

matplotlib.use('Agg')  # render without display (headless execution)
import matplotlib.pyplot as plt  # plotting API

plt.style.use('seaborn-v0_8')  # consistent figure style


def main() -> None:
    out = Path('figures/ch12_cosine_schedule.svg')  # SVG output path
    out.parent.mkdir(parents=True, exist_ok=True)  # ensure figures directory exists

    epochs = np.arange(0, 100)  # epoch indices for plotting
    base_lr = 5e-3  # baseline learning rate shared by both schedules
    constant = np.full_like(epochs, base_lr, dtype=float)  # constant schedule baseline
    cosine = 0.5 * base_lr * (1 + np.cos(np.pi * epochs / epochs.max()))  # cosine annealing curve

    fig, ax = plt.subplots(figsize=(6.0, 3.6))  # create figure and axes
    ax.plot(epochs, constant, label='Constant LR', alpha=0.9)  # plot constant line
    ax.plot(epochs, cosine, label='Cosine schedule', alpha=0.9)  # plot cosine curve
    ax.set_xlabel('Epoch')  # label x-axis
    ax.set_ylabel('Learning rate')  # label y-axis
    ax.grid(True, alpha=0.3)  # add light gridlines
    ax.legend(frameon=False)  # legend without frame
    ax.set_title('Cosine vs constant learning rate')  # chart title
    fig.tight_layout()  # minimise whitespace

    fig.savefig(out, format='svg')  # write SVG asset
    png_out = out.with_suffix('.png')  # companion PNG path
    fig.savefig(png_out, dpi=200)  # save PNG for beamer
    print(f"Wrote {out} and {png_out}")  # log outputs


if __name__ == '__main__':  # allow CLI execution
    main()  # generate figure
