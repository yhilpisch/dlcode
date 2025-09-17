"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 13 — Padding/truncation diagram (SVG).

Output: figures/ch13_pad_trunc.svg
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.style.use('seaborn-v0_8')

def row(ax, y, tokens, max_len=8, x0=0.8):
    for i in range(max_len):
        r = Rectangle((x0 + i, y), 1, 0.8, fc='#eef4ff', ec='#446')
        ax.add_patch(r)
        txt = tokens[i] if i < len(tokens) else '<pad>'
        ax.text(x0 + i + 0.5, y+0.4, txt, ha='center', va='center', fontsize=9)

def main() -> None:
    out = Path('figures/ch13_pad_trunc.svg'); out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0,2.8)); ax.axis('off')
    x0 = 0.9; max_len = 8
    ax.set_xlim(0.0, x0 + max_len + 0.2); ax.set_ylim(0, 3)
    row(ax, 2.0, ['good','movie'], max_len=max_len, x0=x0)
    row(ax, 1.0, ['this','movie','is','good','!','indeed','great'], max_len=max_len, x0=x0)
    row(ax, 0.0, ['bad'], max_len=max_len, x0=x0)
    # Row labels with margin to avoid overlap
    ax.text(0.2, 2.35, 'Pad', ha='left', va='bottom')
    ax.text(0.2, 1.35, 'Truncate', ha='left', va='bottom')
    ax.text(0.2, 0.35, 'Pad', ha='left', va='bottom')
    fig.tight_layout(); fig.savefig(out, format='svg'); print(f"Wrote {out}")

if __name__ == '__main__':
    main()
