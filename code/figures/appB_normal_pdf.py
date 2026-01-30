"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Appendix B — Generate standard normal PDF SVG.

Output: figures/appB_normal_pdf.svg
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
    out = Path("figures/appB_normal_pdf.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    xs = np.linspace(-4, 4, 400)
    pdf = np.exp(-0.5*xs**2)/np.sqrt(2*np.pi)

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(xs, pdf, color='C1')
    ax.set_xlabel('x'); ax.set_ylabel('density')
    ax.set_title('Standard normal PDF')
    fig.tight_layout(); fig.savefig(out, format='svg')
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

