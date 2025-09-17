"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Appendix D — Tangent line to f(x)=x^2 at x0=1.

Output: figures/appD_tangent_line.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')


def main() -> None:
    out = Path("figures/appD_tangent_line.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    def f(x: np.ndarray) -> np.ndarray:
        return x * x

    x0 = 1.0
    m = 2 * x0  # derivative of x^2
    b = f(x0) - m * x0

    xs = np.linspace(-0.5, 2.5, 400)
    ys = f(xs)
    yt = m * xs + b

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(xs, ys, label='f(x)=x^2', color='C0', lw=2.0)
    ax.plot(xs, yt, label='tangent at x0=1', color='C1', lw=2.0)
    ax.scatter([x0], [f(x0)], color='k', s=30)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

