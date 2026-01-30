"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 6 — Simple neuron schematic (SVG).

Output: figures/ch06_neuron_diagram.svg
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from code.figures._save import save_png_pdf
plt.style.use('seaborn-v0_8')


def main() -> None:
    out = Path("figures/ch06_neuron_diagram.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    # Inputs
    for i, y in enumerate([3.2, 2.0, 0.8]):
        ax.plot([0.6, 3.0], [y, 2.0], color='#777')
        ax.text(
            0.25,
            y,
            f"x{i+1}",
            ha="right",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.3),
        )
        ax.text(
            1.8,
            (y + 2.0) / 2 + 0.25,
            f"w{i+1}",
            fontsize=10,
            color="#333",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.2),
        )
    # Neuron body
    circ = plt.Circle((3.2, 2.0), 0.5, color='#e0ffe0', ec='#555')
    ax.add_patch(circ)
    ax.text(3.2, 2.0, '∑', ha='center', va='center', fontsize=12)
    ax.text(2.8, 1.2, '+ b', fontsize=10)
    # Activation
    ax.annotate(
        "",
        xy=(5.0, 2.0),
        xytext=(3.7, 2.0),
        arrowprops=dict(arrowstyle="->"),
    )
    ax.text(
        4.1,
        2.3,
        r"$\phi$",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2),
    )
    ax.annotate(
        "",
        xy=(8.8, 2.0),
        xytext=(5.2, 2.0),
        arrowprops=dict(arrowstyle="->"),
    )
    ax.text(9.0, 2.0, 'y', ha='left', va='center')
    fig.tight_layout()
    fig.savefig(out, format="svg")
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
