"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 9 — Data pipeline diagram (SVG).

Output: figures/ch09_pipeline_diagram.svg
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from code.figures._save import save_png_pdf
plt.style.use('seaborn-v0_8')


def box(ax, xy, w, h, text):
    rect = plt.Rectangle(
        (xy[0], xy[1]),
        w,
        h,
        fc="#eef4ff",
        ec="#446",
        lw=1.0,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
    )


def arrow(ax, p, q):
    ax.annotate(
        "",
        xy=q,
        xytext=p,
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"),
    )


def main() -> None:
    out = Path("figures/ch09_pipeline_diagram.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 2.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)

    box(ax, (0.5, 1.0), 1.6, 1.0, 'Raw data\n(NP arrays, files)')
    box(ax, (3.0, 1.0), 1.8, 1.0, 'Dataset\n(__getitem__/__len__)')
    box(ax, (5.6, 1.0), 1.8, 1.0, 'DataLoader\n(batches, shuffle)')
    box(ax, (8.2, 1.0), 1.3, 1.0, 'Model\n+ Loss')

    arrow(ax, (2.1, 1.5), (3.0, 1.5))
    arrow(ax, (4.8, 1.5), (5.6, 1.5))
    arrow(ax, (7.4, 1.5), (8.2, 1.5))

    fig.tight_layout()
    fig.savefig(out, format="svg")
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
