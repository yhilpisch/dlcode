"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    out = Path("figures/ch17_project_scope.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.set_xlabel("Estimated Effort (weeks)")
    ax.set_ylabel("Impact (portfolio/learning)")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 10)
    ax.grid(alpha=0.2)

    # Sample project points
    projects = [
        (1.0, 5.5, "Baseline + Card"),
        (2.0, 7.0, "Aug + Analysis"),
        (3.0, 8.5, "LoRA Demo"),
        (4.5, 9.0, "End-to-End App"),
        (0.8, 3.5, "Tiny Repro"),
    ]
    for x, y, label in projects:
        ax.scatter([x], [y], s=50, color="#1f77b4")
        ax.text(x + 0.08, y + 0.1, label, fontsize=9)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")


if __name__ == "__main__":
    main()

