"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 12 — Throughput vs batch size (SVG).

Output: figures/ch12_throughput_batch.svg
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

def main() -> None:
    out = Path('figures/ch12_throughput_batch.svg'); out.parent.mkdir(parents=True, exist_ok=True)
    bs = np.array([8, 16, 32, 64, 128, 256])
    # Toy curve: increases and plateaus
    thr = 800 * (1 - np.exp(-bs/64.0)) + 100  # samples/sec
    fig, ax = plt.subplots(figsize=(6.4,3.6))
    ax.plot(bs, thr, marker='o'); ax.set_xscale('log', base=2)
    ax.set_xlabel('Batch size'); ax.set_ylabel('Samples/sec'); ax.grid(True, alpha=0.3)
    ax.set_title('Throughput vs batch size (toy)')
    for x,y in zip(bs, thr): ax.text(x, y+8, f"{int(y)}", ha='center', fontsize=9)
    fig.tight_layout(); fig.savefig(out, format='svg'); print(f"Wrote {out}")

if __name__ == '__main__':
    main()

