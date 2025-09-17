"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 12 — Cosine vs constant LR loss curves (SVG).

Output: figures/ch12_cosine_schedule.svg
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

def main() -> None:
    out = Path('figures/ch12_cosine_schedule.svg'); out.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(0, 100)
    base = 1.0/(1+0.05*epochs) + 0.02*np.random.RandomState(0).randn(len(epochs))
    const = base - 0.02
    cos = (1.0+np.cos(np.pi*epochs/100.0))/2
    loss_cos = 0.4*base* (0.5+0.5*cos)
    fig, ax = plt.subplots(figsize=(6.0,3.6))
    ax.plot(epochs, const, label='Constant LR', alpha=0.9)
    ax.plot(epochs, loss_cos, label='Cosine schedule', alpha=0.9)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (toy)'); ax.grid(True, alpha=0.3)
    ax.legend(frameon=False); ax.set_title('Cosine vs constant LR (toy run)')
    fig.tight_layout(); fig.savefig(out, format='svg'); print(f"Wrote {out}")

if __name__ == '__main__':
    main()

