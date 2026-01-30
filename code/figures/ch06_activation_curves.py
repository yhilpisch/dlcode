"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 6 — Activation function curves (SVG).

Output: figures/ch06_activation_curves.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from code.figures._save import save_png_pdf
plt.style.use('seaborn-v0_8')


def main() -> None:
    out = Path("figures/ch06_activation_curves.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-6, 6, 400)
    sigmoid = 1/(1+np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0, x)
    plt.figure(figsize=(5.2, 3.4))
    plt.plot(x, sigmoid, label='sigmoid')
    plt.plot(x, tanh, label='tanh')
    plt.plot(x, relu, label='ReLU')
    plt.xlabel('x'); plt.ylabel('activation'); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out, format='svg')
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

