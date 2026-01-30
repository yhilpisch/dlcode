"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 12 — Checkpoint contents/flow diagram (SVG).

Output: figures/ch12_checkpoints.svg
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from code.figures._save import save_png_pdf
plt.style.use("seaborn-v0_8")


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
) -> None:
    """Draw a labelled rectangle at the given coordinates."""
    rect = Rectangle((x, y), w, h, fc="#eef4ff", ec="#446")
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center")


def main() -> None:
    out = Path("figures/ch12_checkpoints.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.axis("off")

    left_x, mid_x, right_x = 0.2, 2.8, 5.0
    box_width, box_height = 1.8, 0.6
    y_levels = [1.8, 1.0, 0.2]

    box(ax, left_x, y_levels[0], box_width, box_height, "model.state_dict()")
    box(ax, left_x, y_levels[1], box_width, box_height, "opt.state_dict()")
    box(ax, left_x, y_levels[2], box_width, box_height, "sched.state_dict()")

    box(ax, mid_x, 1.0, box_width, box_height, "checkpoint.pt")
    box(ax, mid_x, 0.2, box_width, box_height, "epoch, RNG state")

    right_labels = [
        "model.load_state_dict(...)",
        "opt.load_state_dict(...)",
        "sched.load_state_dict(...)",
        "training loop (resume)",
    ]
    right_start = 2.0
    box_spacing = 0.8
    right_y = [
        right_start - i * box_spacing for i in range(len(right_labels))
    ]
    for label, y in zip(right_labels, right_y):
        box(ax, right_x, y, box_width, box_height, label)

    mid_center_y = 1.0 + box_height / 2
    for y in y_levels:
        source_center_y = y + box_height / 2
        arrow = FancyArrow(
            left_x + box_width,
            source_center_y,
            mid_x - (left_x + box_width) - 0.1,
            mid_center_y - source_center_y,
            width=0.001,
            head_width=0.08,
            length_includes_head=True,
            color="black",
        )
        ax.add_patch(arrow)

    for y in right_y[:3]:
        arrow = FancyArrow(
            mid_x + box_width,
            1.0 + box_height / 2,
            right_x - (mid_x + box_width) - 0.12,
            (y + box_height / 2) - (1.0 + box_height / 2),
            width=0.001,
            head_width=0.08,
            length_includes_head=True,
            color="black",
        )
        ax.add_patch(arrow)

    loop_y = right_y[3]

    arrow = FancyArrow(
        mid_x + box_width,
        0.2 + box_height / 2,
        right_x - (mid_x + box_width) - 0.12,
        (loop_y + box_height / 2) - (0.2 + box_height / 2),
        width=0.001,
        head_width=0.08,
        length_includes_head=True,
        color="black",
    )
    ax.add_patch(arrow)

    ax.set_xlim(0.0, 7.2)
    ax.set_ylim(-0.7, 2.6)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    png_out = out.with_suffix(".png")
    fig.savefig(png_out, dpi=200)
    save_png_pdf(out)
    print(f"Wrote {out} and {png_out}")


if __name__ == "__main__":
    main()
