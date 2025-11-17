"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 12 — AMP vs FP32 relative speed (SVG).

Output: figures/ch12_amp_speed.svg
"""
from __future__ import annotations  # future annotations for Python <3.11 compatibility
from pathlib import Path  # filesystem path helpers for output directory
import numpy as np  # numerical arrays for bar heights
import matplotlib  # base Matplotlib interface
matplotlib.use('Agg')  # headless backend suitable for scripted figure generation
import matplotlib.pyplot as plt  # plotting API
plt.style.use('seaborn-v0_8')  # consistent styling across figures


def main() -> None:
    out = Path('figures/ch12_amp_speed.svg')  # target output path
    out.parent.mkdir(parents=True, exist_ok=True)  # ensure figures/ directory exists
    backends = ['CPU', 'GPU']  # categories shown on the x-axis
    fp32 = np.array([1.0, 1.0])  # baseline throughput for FP32 runs
    amp = np.array([1.0, 1.6])  # relative throughput when AMP is enabled
    positions = np.arange(len(backends))  # bar positions for grouped chart
    bar_width = 0.35  # width of each bar within group
    fig, ax = plt.subplots(figsize=(5.2, 3.6))  # create figure and axes
    ax.bar(positions - bar_width / 2, fp32, width=bar_width, label='FP32')  # plot FP32 bars
    ax.bar(positions + bar_width / 2, amp, width=bar_width, label='AMP (FP16/BF16)')  # plot AMP bars
    ax.set_xticks(positions)  # set tick locations
    ax.set_xticklabels(backends)  # label ticks with backend names
    ax.set_ylabel('Relative throughput')  # y-axis label
    ax.set_ylim(0, 1.9)  # fix y-axis range for comparability
    ax.legend(frameon=False)  # place legend without frame
    ax.grid(True, axis='y', alpha=0.3)  # add horizontal gridlines for readability
    ax.set_title('AMP vs FP32 (relative)')  # figure title
    fig.tight_layout()  # minimize excess whitespace
    fig.savefig(out, format='svg')  # write SVG asset to disk
    print(f"Wrote {out}")  # confirm output path on stdout


if __name__ == '__main__':  # allow running as a script
    main()  # generate figure
