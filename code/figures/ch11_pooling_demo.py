"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 11 — Pooling 2×2 stride 2 demo (SVG).

Output: figures/ch11_pooling_demo.svg
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

def main() -> None:
    out=Path('figures/ch11_pooling_demo.svg'); out.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(1,17).reshape(4,4)
    fig, axes = plt.subplots(1,2, figsize=(6.2,2.6))
    im0=axes[0].imshow(x, cmap='viridis'); axes[0].set_title('input 4×4'); axes[0].axis('off')
    pooled = x.reshape(2,2,2,2).max(axis=(2,3))
    im1=axes[1].imshow(pooled, cmap='viridis'); axes[1].set_title('max pool 2×2 → 2×2'); axes[1].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out, format='svg'); print(f"Wrote {out}")
if __name__=='__main__': main()
