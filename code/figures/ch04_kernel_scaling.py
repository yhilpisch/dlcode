"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 4 — Kernel method scaling illustration (SVG).

Output: figures/ch04_kernel_scaling.svg
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
    out = Path("figures/ch04_kernel_scaling.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    n = np.array([200, 500, 1_000, 2_000, 5_000, 10_000])
    # Kernel matrix memory ~ n^2 entries (double precision ~8 bytes)
    mem_gb = (n.astype(float) ** 2 * 8) / (1024 ** 3)

    plt.figure(figsize=(5.0, 3.2))
    plt.plot(n, mem_gb, marker='o')
    plt.xlabel('samples n'); plt.ylabel('Kernel matrix memory (GB)')
    plt.title('Kernel methods scale poorly with n (memory ~ n^2)')
    plt.tight_layout(); plt.savefig(out, format='svg')
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

