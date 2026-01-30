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
    out = Path("figures/ch16_fairness_bars.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    groups = ["Group A", "Group B"]
    tpr = np.array([0.92, 0.82])
    fpr = np.array([0.08, 0.14])

    x = np.arange(len(groups))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x - w/2, tpr, w, label="TPR", color="#2ca02c")
    ax.bar(x + w/2, fpr, w, label="FPR", color="#d62728")
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Group metrics (illustrative)")
    ax.legend(loc="upper right")
    for xi, yi in zip(x - w/2, tpr):
        ax.text(xi, yi + 0.02, f"{yi:.2f}", ha="center", fontsize=9)
    for xi, yi in zip(x + w/2, fpr):
        ax.text(xi, yi + 0.02, f"{yi:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    save_png_pdf(out)


if __name__ == "__main__":
    main()

