"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 11 — Convolution sliding demo

Outputs:
- figures/ch11_conv_slide.svg
- figures/ch11_conv_slide.png

Goal: Make the output map actually reflect the kernel response over the toy
image so that edges/patterns visibly "light up" in the figure.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from code.figures._save import save_png_pdf
plt.style.use('seaborn-v0_8')

def conv2d_valid(x: np.ndarray, k: np.ndarray, stride: int = 1) -> np.ndarray:
    """Naive valid 2D convolution (no padding).

    x: (H, W), k: (KH, KW), stride >= 1
    returns y of shape ((H-KH)//stride+1, (W-KW)//stride+1)
    """
    H, W = x.shape
    KH, KW = k.shape
    oh = (H - KH) // stride + 1
    ow = (W - KW) // stride + 1
    y = np.empty((oh, ow), dtype=float)
    for i in range(oh):
        for j in range(ow):
            ii = i * stride
            jj = j * stride
            y[i, j] = np.sum(x[ii:ii+KH, jj:jj+KW] * k)
    return y


def main() -> None:
    out_svg = Path('figures/ch11_conv_slide.svg')
    out_png = Path('figures/ch11_conv_slide.png')
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    # Toy image: black background with a white 2x2 square in the middle
    img = np.zeros((6, 6))
    img[2:4, 2:4] = 1.0

    # Edge-emphasizing kernel (discrete Laplacian-like cross)
    ker = np.array([[0., 1., 0.],
                    [1., -4., 1.],
                    [0., 1., 0.]])

    y = conv2d_valid(img, ker, stride=1)

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    # Input
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('input 6×6')
    axes[0].axis('off')
    # Kernel
    vmax_k = np.max(np.abs(ker))
    axes[1].imshow(ker, cmap='coolwarm', vmin=-vmax_k, vmax=vmax_k)
    axes[1].set_title('kernel 3×3')
    axes[1].axis('off')
    # Output: show signed responses with symmetric limits so edges are visible
    vmax_y = float(np.max(np.abs(y))) or 1.0
    axes[2].imshow(y, cmap='coolwarm', vmin=-vmax_y, vmax=vmax_y)
    axes[2].set_title('output 4×4')
    axes[2].axis('off')

    fig.tight_layout()
    fig.savefig(out_svg, format='svg')
    fig.savefig(out_png, dpi=200)
    save_png_pdf(out_svg)
    print(f"Wrote {out_svg} and {out_png}")
if __name__=='__main__': main()
