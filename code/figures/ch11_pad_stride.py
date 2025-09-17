"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 11 — Padding/stride output size demo (SVG).

Output: figures/ch11_pad_stride.svg
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

def conv2d(x: np.ndarray, k: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    if padding > 0:
        x = np.pad(x, ((padding, padding), (padding, padding)), mode='constant')
    H, W = x.shape
    KH, KW = k.shape
    out_h = (H - KH) // stride + 1
    out_w = (W - KW) // stride + 1
    y = np.empty((out_h, out_w), dtype=float)
    for i in range(out_h):
        for j in range(out_w):
            ii = i * stride
            jj = j * stride
            y[i, j] = np.sum(x[ii:ii+KH, jj:jj+KW] * k)
    return y

def board(n):
    return np.arange(n*n).reshape(n,n)

def _show_out(ax, arr: np.ndarray, title: str) -> None:
    h, w = arr.shape
    im = ax.imshow(arr, cmap='viridis', interpolation='nearest')
    ax.set_title(f"{title} — {h}×{w}", fontsize=10)
    # Draw gridlines to make size visually obvious
    ax.set_xticks(np.arange(w))
    ax.set_yticks(np.arange(h))
    ax.set_xticks(np.arange(w+1)-0.5, minor=True)
    ax.set_yticks(np.arange(h+1)-0.5, minor=True)
    ax.grid(which='minor', color='#ffffff', linewidth=1.0)
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)

def main() -> None:
    out=Path('figures/ch11_pad_stride.svg'); out.parent.mkdir(parents=True, exist_ok=True)
    # Input 5x5 and 3x3 averaging kernel
    x = board(5).astype(float)
    k = np.ones((3,3), dtype=float) / 9.0
    cases = [
        (0, 1, 'p0,s1'),
        (1, 1, 'p1,s1'),
        (0, 2, 'p0,s2'),
    ]
    fig, axes = plt.subplots(1,3, figsize=(7.6,2.6))
    for ax, (p, s, title) in zip(axes, cases):
        y = conv2d(x, k, stride=s, padding=p)
        _show_out(ax, y, title)
    fig.tight_layout(); fig.savefig(out, format='svg'); print(f"Wrote {out}")
if __name__=='__main__': main()
