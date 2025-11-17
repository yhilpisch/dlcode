"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 12 — Checkpoint contents/flow diagram (SVG).

Output: figures/ch12_checkpoints.svg
"""
from __future__ import annotations  # enable postponed annotations
from pathlib import Path  # filesystem helpers for output path
import matplotlib  # base Matplotlib import
matplotlib.use('Agg')  # headless backend for script execution
import matplotlib.pyplot as plt  # plotting API
from matplotlib.patches import Rectangle, FancyArrow  # shape primitives
plt.style.use('seaborn-v0_8')  # consistent styling across figures


def box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str) -> None:
    """Draw a labelled rectangle at the given coordinates."""
    rect = Rectangle((x, y), w, h, fc='#eef4ff', ec='#446')  # shaded box with border
    ax.add_patch(rect)  # add rectangle to the axes
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center')  # centre text inside box


def main() -> None:
    out = Path('figures/ch12_checkpoints.svg')  # destination SVG path
    out.parent.mkdir(parents=True, exist_ok=True)  # ensure figures directory exists
    fig, ax = plt.subplots(figsize=(7.2, 2.8))  # create drawing canvas
    ax.axis('off')  # hide axes for clean diagram

    left_x, mid_x, right_x = 0.2, 2.8, 5.0  # column x-coordinates
    box_width, box_height = 1.8, 0.6  # uniform box dimensions
    y_levels = [1.8, 1.0, 0.2]  # vertical positions for stacked boxes on the left

    box(ax, left_x, y_levels[0], box_width, box_height, 'model.state_dict()')  # model weights box
    box(ax, left_x, y_levels[1], box_width, box_height, 'opt.state_dict()')  # optimizer state box
    box(ax, left_x, y_levels[2], box_width, box_height, 'sched.state_dict()')  # scheduler state box

    box(ax, mid_x, 1.0, box_width, box_height, 'checkpoint.pt')  # core checkpoint file
    box(ax, mid_x, 0.2, box_width, box_height, 'epoch, RNG state')  # auxiliary metadata

    right_labels = [
        'model.load_state_dict(...)',
        'opt.load_state_dict(...)',
        'sched.load_state_dict(...)',
        'training loop (resume)',
    ]
    right_start = 2.0  # y-position of the topmost box on the right
    box_spacing = 0.8  # vertical spacing between box origins
    right_y = [right_start - i * box_spacing for i in range(len(right_labels))]  # evenly spaced positions
    for label, y in zip(right_labels, right_y):
        box(ax, right_x, y, box_width, box_height, label)

    mid_center_y = 1.0 + box_height / 2  # vertical center of checkpoint box
    for y in y_levels:  # arrows from state boxes to checkpoint file
        source_center_y = y + box_height / 2  # current box center
        arrow = FancyArrow(left_x + box_width, source_center_y,
                           mid_x - (left_x + box_width) - 0.1,
                           mid_center_y - source_center_y,
                           width=0.001, head_width=0.08, length_includes_head=True, color='black')
        ax.add_patch(arrow)  # add arrow patch

    for y in right_y[:3]:  # arrows from checkpoint file to restore calls
        arrow = FancyArrow(mid_x + box_width, 1.0 + box_height / 2,
                           right_x - (mid_x + box_width) - 0.12,
                           (y + box_height / 2) - (1.0 + box_height / 2),
                           width=0.001, head_width=0.08, length_includes_head=True, color='black')
        ax.add_patch(arrow)  # add arrow patch

    loop_y = right_y[3]  # y-position for training loop box (already drawn)

    arrow = FancyArrow(mid_x + box_width, 0.2 + box_height / 2,
                       right_x - (mid_x + box_width) - 0.12,
                       (loop_y + box_height / 2) - (0.2 + box_height / 2),
                       width=0.001, head_width=0.08, length_includes_head=True, color='black')
    ax.add_patch(arrow)  # arrow from metadata to resume loop

    ax.set_xlim(0.0, 7.2)  # horizontal bounds for layout
    ax.set_ylim(-0.7, 2.6)  # vertical bounds for layout
    fig.tight_layout()  # minimize padding
    fig.savefig(out, format='svg')  # write SVG figure
    png_out = out.with_suffix('.png')  # companion PNG for beamer
    fig.savefig(png_out, dpi=200)  # save high-resolution PNG
    print(f"Wrote {out} and {png_out}")  # log output locations


if __name__ == '__main__':  # allow running as a script
    main()  # render diagram
