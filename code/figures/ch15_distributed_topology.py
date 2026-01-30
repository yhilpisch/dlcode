"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from code.figures._save import save_png_pdf


def main() -> None:
    out = Path("figures/ch15_distributed_topology.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.axis("off")

    # Draw 4 ranks and GPUs
    for i in range(4):
        x = 0.1 + i * 0.2
        # GPU box
        ax.add_patch(plt.Rectangle((x, 0.3), 0.16, 0.5, ec="#444", fc="#e6f2ff"))
        ax.text(x + 0.08, 0.55, f"GPU{i}", ha="center", va="center", fontsize=10)
        # Rank label
        ax.text(x + 0.08, 0.25, f"rank {i}", ha="center", va="center", fontsize=9)

    # Allreduce ring arrows (schematic)
    xs = [0.18, 0.38, 0.58, 0.78, 0.18]
    ys = [0.75, 0.75, 0.75, 0.75, 0.75]
    ax.plot(xs, ys, color="#1f77b4", lw=1.5, alpha=0.9)
    for i in range(4):
        ax.annotate("", xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#1f77b4"))
    ax.text(0.5, 0.85, "Gradient allreduce (NCCL)", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    save_png_pdf(out)


if __name__ == "__main__":
    main()

