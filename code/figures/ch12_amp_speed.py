"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 12 — AMP vs FP32 relative speed (SVG).

Output: figures/ch12_amp_speed.svg
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

def main() -> None:
    out = Path('figures/ch12_amp_speed.svg'); out.parent.mkdir(parents=True, exist_ok=True)
    backends = ['CPU', 'GPU']
    fp32 = np.array([1.0, 1.0])
    amp  = np.array([1.0, 1.6])
    x = np.arange(len(backends))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.2,3.6))
    ax.bar(x - w/2, fp32, width=w, label='FP32')
    ax.bar(x + w/2, amp,  width=w, label='AMP (FP16/BF16)')
    ax.set_xticks(x); ax.set_xticklabels(backends)
    ax.set_ylabel('Relative throughput'); ax.set_ylim(0, 1.9)
    ax.legend(frameon=False); ax.grid(True, axis='y', alpha=0.3)
    ax.set_title('AMP vs FP32 (relative)')
    fig.tight_layout(); fig.savefig(out, format='svg'); print(f"Wrote {out}")

if __name__ == '__main__':
    main()

