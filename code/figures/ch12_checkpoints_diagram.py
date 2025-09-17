"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 12 — Checkpoint contents/flow diagram (SVG).

Output: figures/ch12_checkpoints.svg
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
plt.style.use('seaborn-v0_8')

def box(ax, x, y, w, h, text):
    r = Rectangle((x,y), w, h, fc='#eef4ff', ec='#446'); ax.add_patch(r)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center')

def main() -> None:
    out = Path('figures/ch12_checkpoints.svg'); out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2,2.8)); ax.axis('off')
    # Layout coordinates (x,y,w,h)
    left_x, mid_x, right_x = 0.2, 2.8, 5.0
    w, h = 1.8, 0.6
    ys = [1.8, 1.0, 0.2]
    # Left: states to save
    box(ax, left_x, ys[0], w, h, 'model.state_dict()')
    box(ax, left_x, ys[1], w, h, 'opt.state_dict()')
    box(ax, left_x, ys[2], w, h, 'sched.state_dict()')
    # Middle: file with extras
    box(ax, mid_x, 1.0, w, h, 'checkpoint.pt')
    # Extras below file: epoch and RNG
    box(ax, mid_x, 0.2, w, h, 'epoch, RNG state')
    # Right: restore calls
    box(ax, right_x, ys[0], w, h, 'model.load_state_dict(...)')
    box(ax, right_x, ys[1], w, h, 'opt.load_state_dict(...)')
    box(ax, right_x, ys[2], w, h, 'sched.load_state_dict(...)')
    # Arrows: left -> checkpoint
    for y in ys:
        ax.add_patch(FancyArrow(left_x+w, y+h/2, mid_x-(left_x+w)-0.1, 0, width=0.001,
                                head_width=0.08, length_includes_head=True, color='black'))
    # Arrows: checkpoint -> right (from center of checkpoint to centers of targets)
    for y in ys:
        ax.add_patch(FancyArrow(mid_x+w, 1.0+h/2, (right_x - (mid_x+w) - 0.12), (y + h/2) - (1.0 + h/2),
                                width=0.001, head_width=0.08, length_includes_head=True, color='black'))
    # Training loop box placed below to avoid overlap with right column
    loop_y = -0.3
    box(ax, right_x, loop_y, w, h, 'training loop (resume)')
    # Arrow: epoch/RNG -> training loop (from right edge center of epoch box)
    ax.add_patch(FancyArrow(mid_x + w, 0.2 + h/2, (right_x - (mid_x + w) - 0.12), (loop_y + h/2) - (0.2 + h/2),
                            width=0.001, head_width=0.08, length_includes_head=True, color='black'))
    # Set bounds to avoid clipping
    ax.set_xlim(0, 7.2); ax.set_ylim(-0.7, 2.6)
    fig.tight_layout(); fig.savefig(out, format='svg'); print(f"Wrote {out}")

if __name__ == '__main__':
    main()
