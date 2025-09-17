"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 10 — Learning-rate schedules visualization (SVG).

Output: figures/ch10_lr_schedules.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')


def main() -> None:
    out = Path('figures/ch10_lr_schedules.svg'); out.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(0, 100)
    lr_const = np.full_like(epochs, 5e-3, dtype=float)
    lr_step = np.where(epochs < 60, 5e-3, 2.5e-3)
    lr_cos = 0.5*(1+np.cos(np.pi*epochs/100))*5e-3
    plt.figure(figsize=(6.4,3.0))
    plt.plot(epochs, lr_const, label='constant')
    plt.plot(epochs, lr_step, label='step@60')
    plt.plot(epochs, lr_cos, label='cosine')
    plt.xlabel('epoch'); plt.ylabel('learning rate'); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out, format='svg')
    print(f"Wrote {out}")

if __name__ == '__main__':
    main()
