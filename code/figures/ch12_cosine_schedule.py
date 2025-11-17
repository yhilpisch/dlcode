"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 12 — Cosine vs constant LR loss curves (SVG).

Output: figures/ch12_cosine_schedule.svg
"""
from __future__ import annotations  # postponed evaluation of annotations
from pathlib import Path  # filesystem utilities
import numpy as np  # numerical routines for synthetic data
import matplotlib  # base Matplotlib import
matplotlib.use('Agg')  # render without display (CLI friendly)
import matplotlib.pyplot as plt  # plotting API
plt.style.use('seaborn-v0_8')  # consistent figure style


def main() -> None:
    out = Path('figures/ch12_cosine_schedule.svg')  # output SVG path
    out.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    epochs = np.arange(0, 100)  # epoch indices for x-axis
    rng = np.random.RandomState(0)  # seeded RNG for reproducibility
    base = 1.0 / (1 + 0.05 * epochs) + 0.02 * rng.randn(len(epochs))  # toy decaying loss baseline
    const_loss = base - 0.02  # simulated constant-LR trajectory
    cosine_phase = (1.0 + np.cos(np.pi * epochs / 100.0)) / 2  # cosine modulation between 0 and 1
    cosine_loss = 0.4 * base * (0.5 + 0.5 * cosine_phase)  # toy cosine-scheduled loss
    fig, ax = plt.subplots(figsize=(6.0, 3.6))  # create figure
    ax.plot(epochs, const_loss, label='Constant LR', alpha=0.9)  # plot constant schedule curve
    ax.plot(epochs, cosine_loss, label='Cosine schedule', alpha=0.9)  # plot cosine schedule curve
    ax.set_xlabel('Epoch')  # label x-axis
    ax.set_ylabel('Loss (toy)')  # label y-axis
    ax.grid(True, alpha=0.3)  # add light grid for readability
    ax.legend(frameon=False)  # show legend without frame
    ax.set_title('Cosine vs constant LR (toy run)')  # set plot title
    fig.tight_layout()  # reduce whitespace
    fig.savefig(out, format='svg')  # save figure to disk
    print(f"Wrote {out}")  # log output location


if __name__ == '__main__':  # allow CLI invocation
    main()  # generate figure
