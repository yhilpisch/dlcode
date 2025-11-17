"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 12 — Gradient accumulation vs memory (SVG).

Output: figures/ch12_grad_accum.svg
"""
from __future__ import annotations  # postponed annotations for compatibility
from pathlib import Path  # filesystem utilities for output path
import numpy as np  # numerical helpers for toy data
import matplotlib  # base Matplotlib import
matplotlib.use('Agg')  # headless backend suitable for scripts
import matplotlib.pyplot as plt  # plotting API
plt.style.use('seaborn-v0_8')  # consistent style across figures


def main() -> None:
    out = Path('figures/ch12_grad_accum.svg')  # destination SVG asset
    out.parent.mkdir(parents=True, exist_ok=True)  # ensure output directory exists
    accumulation = np.array([1, 2, 4, 8])  # accumulation factors under test
    memory = 6.0 / accumulation + 2.0  # toy per-step memory usage in GB
    fig, ax = plt.subplots(figsize=(6.0, 3.6))  # create figure
    ax.plot(accumulation, memory, marker='o')  # plot memory vs accumulation
    ax.set_xscale('log', base=2)  # log-scale x-axis for powers of two
    ax.set_xlabel('Accumulation factor (micro-batches)')  # label x-axis
    ax.set_ylabel('Per-step memory (GB, toy)')  # label y-axis
    ax.grid(True, alpha=0.3)  # add light gridlines
    ax.set_title('Gradient accumulation reduces per-step memory')  # chart title
    for x_value, y_value in zip(accumulation, memory):  # annotate each point
        ax.text(x_value, y_value + 0.15, f"{y_value:.1f}", ha='center', fontsize=9)  # place value label
    fig.tight_layout()  # minimize padding
    fig.savefig(out, format='svg')  # save figure to disk
    print(f"Wrote {out}")  # log output path


if __name__ == '__main__':  # allow script execution from CLI
    main()  # generate diagram
