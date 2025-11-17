"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 12 — Throughput vs batch size (SVG).

Output: figures/ch12_throughput_batch.svg
"""
from __future__ import annotations  # postponed annotations for compatibility
from pathlib import Path  # filesystem utilities for output path
import numpy as np  # numerical helpers for toy data
import matplotlib  # base Matplotlib import
matplotlib.use('Agg')  # headless backend for scripted renders
import matplotlib.pyplot as plt  # plotting API
plt.style.use('seaborn-v0_8')  # consistent house style


def main() -> None:
    out = Path('figures/ch12_throughput_batch.svg')  # destination SVG path
    out.parent.mkdir(parents=True, exist_ok=True)  # ensure output directory exists
    batch_sizes = np.array([8, 16, 32, 64, 128, 256])  # toy batch sizes to plot
    throughput = 800 * (1 - np.exp(-batch_sizes / 64.0)) + 100  # synthetic throughput curve
    fig, ax = plt.subplots(figsize=(6.4, 3.6))  # set up figure and axes
    ax.plot(batch_sizes, throughput, marker='o')  # plot samples/sec vs batch size
    ax.set_xscale('log', base=2)  # log-scale to highlight doubling behaviour
    ax.set_xlabel('Batch size')  # label x-axis
    ax.set_ylabel('Samples/sec')  # label y-axis
    ax.grid(True, alpha=0.3)  # add light grid for readability
    ax.set_title('Throughput vs batch size (toy)')  # chart title
    for x_value, y_value in zip(batch_sizes, throughput):  # annotate each point
        ax.text(x_value, y_value + 8, f"{int(y_value)}", ha='center', fontsize=9)  # show numerical throughput
    fig.tight_layout()  # minimize whitespace
    fig.savefig(out, format='svg')  # persist figure as SVG
    print(f"Wrote {out}")  # log output path


if __name__ == '__main__':  # allow CLI invocation
    main()  # generate figure
