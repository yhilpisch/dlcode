"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Appendix D — Finite-difference derivative error (forward vs central) on sin(x0).

Output: figures/appD_fd_error.svg
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
    out = Path("figures/appD_fd_error.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    x0 = 1.0
    true = np.cos(x0)  # d/dx sin(x) at x0
    hs = np.logspace(-8, -1, 40)  # 1e-8 .. 1e-1
    fwd_err = []
    cen_err = []
    for h in hs:
        fwd = (np.sin(x0 + h) - np.sin(x0)) / h
        cen = (np.sin(x0 + h) - np.sin(x0 - h)) / (2 * h)
        fwd_err.append(abs(fwd - true))
        cen_err.append(abs(cen - true))

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.loglog(hs, fwd_err, label='forward (O(h))', color='C0')
    ax.loglog(hs, cen_err, label='central (O(h^2))', color='C1')
    ax.set_xlabel('h'); ax.set_ylabel('abs error')
    ax.legend(frameon=False)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    fig.tight_layout(); fig.savefig(out, format='svg')
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

