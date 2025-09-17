"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    out = Path("figures/ch15_grad_accum_diagram.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    K = 4  # accumulation steps
    replicas = 2
    micro = 32
    eff = K * replicas * micro

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.axis("off")
    # Header row with extra vertical padding to avoid overlaps in PDF
    ax.text(0.02, 0.95, f"micro = {micro}", fontsize=9)
    ax.text(0.28, 0.95, f"accum = {K}", fontsize=9)
    ax.text(0.50, 0.95, f"replicas = {replicas}", fontsize=9)
    ax.text(0.83, 0.95, f"effective = {eff}", fontsize=10, fontweight="bold", ha="right")

    # draw K boxes per replica
    for r in range(replicas):
        y = 0.65 - r * 0.28
        ax.text(0.02, y + 0.06, f"GPU{r}", fontsize=9)
        for i in range(K):
            x = 0.15 + i * 0.12
            ax.add_patch(plt.Rectangle((x, y), 0.1, 0.12, ec="#555", fc="#f2f2f2"))
            ax.text(x + 0.05, y + 0.06, f"{micro}", ha="center", va="center", fontsize=8)
        # Arrow from last micro-batch to a label placed to the right, outside boxes
        ax.annotate("accumulate grads",
                    xy=(0.63, y + 0.06), xytext=(0.76, y + 0.06),
                    arrowprops=dict(arrowstyle="->", lw=1), fontsize=9,
                    ha="left", va="center", clip_on=False)
    # Place step label well below the boxes to avoid overlap in PDF
    ax.text(0.76, 0.30, "optimizer.step()", fontsize=9, ha="center")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")


if __name__ == "__main__":
    main()
