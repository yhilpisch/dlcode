"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 12 — Throughput vs batch size (SVG).

Output: figures/ch12_throughput_batch.svg
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
    out = Path("figures/ch12_throughput_batch.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    batch_sizes = np.array([8, 16, 32, 64, 128, 256])
    throughput = np.array([260, 470, 640, 740, 790, 795])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(batch_sizes, throughput, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Samples/sec")
    ax.set_ylim(0, 850)
    ax.grid(True, alpha=0.3)
    ax.set_title("Throughput vs batch size (toy)")
    for x_value, y_value in zip(batch_sizes, throughput):
        ax.text(
            x_value,
            y_value + 15,
            f"{int(y_value)}",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out, format="svg")
    png_path = out.with_suffix(".png")
    fig.savefig(png_path, dpi=200)
    save_png_pdf(out)
    print(f"Wrote {out} and {png_path}")


if __name__ == "__main__":
    main()
