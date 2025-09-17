"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 5 — Simple autograd computation graph schematic (SVG).

Output: figures/ch05_autograd_graph.svg
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
plt.style.use('seaborn-v0_8')


def node(ax, xy, text):
    c = plt.Circle(xy, 0.32, color='#e8eefc', ec='#444', lw=1.0)
    ax.add_patch(c)
    ax.text(xy[0], xy[1], text, ha='center', va='center', fontsize=11)


def arrow(ax, p, q, text=None):
    ax.annotate('', xy=q, xytext=p, arrowprops=dict(arrowstyle='->', lw=1.2, color='#333'))
    if text:
        mx, my = (p[0]+q[0])/2, (p[1]+q[1])/2
        ax.text(mx, my+0.15, text, ha='center', va='bottom', fontsize=10, color='#333')


def main() -> None:
    out = Path('figures/ch05_autograd_graph.svg')
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 5.2)
    ax.axis('off')

    # Nodes: x, w, b -> y = x·w + b -> L = MSE(y, t)
    x = (1.5, 4.0)
    w = (1.5, 2.8)
    b = (1.5, 1.6)
    y = (6.5, 3.2)
    L = (10.5, 3.2)
    t = (6.5, 1.4)
    for p, lbl in [(x,'x'),(w,'w'),(b,'b'),(y,'y'),(L,'L'),(t,'t')]:
        node(ax, p, lbl)

    # Light captions above nodes (kept away from arrows)
    ax.text(y[0], y[1]+0.9, r"$y = x\cdot w + b$", ha='center', va='bottom', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.2))
    ax.text(L[0], L[1]+0.9, r"$L = \mathrm{MSE}(y,t)$", ha='center', va='bottom', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.2))

    # Helper to draw curved arrow to reduce overlap
    def curved(ax, p, q, color, rad):
        patch = FancyArrowPatch(p, q, arrowstyle='->', mutation_scale=12,
                                connectionstyle=f'arc3,rad={rad}', lw=1.4, color=color)
        ax.add_patch(patch)

    # Forward arrows (blue), curved to avoid overlaps
    blue = '#1f77b4'
    curved(ax, (x[0]+0.4, x[1]-0.1), (y[0]-0.55, y[1]+0.15), blue, 0.25)
    curved(ax, (w[0]+0.4, w[1]),     (y[0]-0.55, y[1]),      blue, 0.00)
    curved(ax, (b[0]+0.4, b[1]+0.1), (y[0]-0.55, y[1]-0.15), blue, -0.25)
    curved(ax, (y[0]+0.55, y[1]),    (L[0]-0.55, L[1]),      blue, 0.00)
    curved(ax, (t[0]+0.55, t[1]+0.05),(L[0]-0.55, L[1]-0.05), blue, 0.18)

    # Backward gradients (red), routed below
    red = '#cc0000'
    curved(ax, (L[0]-0.55, L[1]-0.25), (y[0]+0.55, y[1]-0.25), red, 0.00)
    curved(ax, (y[0]-0.55, y[1]-0.2), (x[0]+0.4, x[1]-0.25), red, -0.28)
    curved(ax, (y[0]-0.55, y[1]-0.3), (w[0]+0.4, w[1]-0.2), red, -0.08)
    curved(ax, (y[0]-0.55, y[1]-0.4), (b[0]+0.4, b[1]-0.1), red, 0.18)

    # Tiny legend
    ax.plot([2.7, 3.4], [0.7, 0.7], color=blue, lw=1.5)
    ax.text(3.5, 0.7, 'forward', va='center', fontsize=10, color=blue)
    ax.plot([4.7, 5.4], [0.7, 0.7], color=red, lw=1.5)
    ax.text(5.5, 0.7, 'backward', va='center', fontsize=10, color=red)

    plt.tight_layout()
    fig.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
