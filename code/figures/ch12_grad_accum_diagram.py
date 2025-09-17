"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 12 — Gradient accumulation vs memory (SVG).

Output: figures/ch12_grad_accum.svg
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

def main() -> None:
    out = Path('figures/ch12_grad_accum.svg'); out.parent.mkdir(parents=True, exist_ok=True)
    accum = np.array([1,2,4,8])
    mem = 6.0/accum + 2.0  # toy per-step GB
    fig, ax = plt.subplots(figsize=(6.0,3.6))
    ax.plot(accum, mem, marker='o')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Accumulation factor (micro-batches)')
    ax.set_ylabel('Per-step memory (GB, toy)')
    ax.grid(True, alpha=0.3)
    ax.set_title('Gradient accumulation reduces per-step memory')
    for x,y in zip(accum, mem): ax.text(x, y+0.15, f"{y:.1f}", ha='center', fontsize=9)
    fig.tight_layout(); fig.savefig(out, format='svg'); print(f"Wrote {out}")

if __name__ == '__main__':
    main()

