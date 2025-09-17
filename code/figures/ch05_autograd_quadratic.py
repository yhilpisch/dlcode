"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 5 — Quadratic with gradient and descent steps (SVG).

Output: figures/ch05_autograd_quadratic.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')


def main() -> None:
    out = Path("figures/ch05_autograd_quadratic.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    f = lambda w: (w - 2.0) ** 2
    df = lambda w: 2.0 * (w - 2.0)
    w0, lr, steps = 4.5, 0.3, 6
    ws = [w0]
    for _ in range(steps):
        w0 = w0 - lr * df(w0)
        ws.append(w0)

    xs = np.linspace(-1, 5, 400)
    plt.figure(figsize=(5.0, 3.4))
    plt.plot(xs, f(xs), label='f(w)=(w-2)^2')
    ys = f(np.array(ws))
    plt.scatter(ws, ys, c=np.arange(len(ws)), cmap='viridis', label='GD steps', zorder=3)
    for w, y in zip(ws, ys):
        m = df(w)
        xx = np.array([w - 0.6, w + 0.6])
        plt.plot(xx, m * (xx - w) + y, color='gray', lw=1, alpha=0.6)
    plt.xlabel('w'); plt.ylabel('f(w)')
    plt.title('Quadratic, gradient, and GD steps')
    plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

