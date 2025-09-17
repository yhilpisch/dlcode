"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 11 — Convolution sliding demo (SVG).

Output: figures/ch11_conv_slide.svg
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

def main() -> None:
    out = Path('figures/ch11_conv_slide.svg'); out.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((6,6)); img[2:4,2:4]=1.0
    ker = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    fig, axes = plt.subplots(1,3, figsize=(7.6,2.6))
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=1); axes[0].set_title('input 6×6'); axes[0].axis('off')
    axes[1].imshow(ker, cmap='coolwarm'); axes[1].set_title('kernel 3×3'); axes[1].axis('off')
    out_shape=(4,4); axes[2].imshow(np.zeros(out_shape), cmap='gray'); axes[2].set_title('output 4×4'); axes[2].axis('off')
    fig.tight_layout(); fig.savefig(out, format='svg'); print(f"Wrote {out}")
if __name__=='__main__': main()
