"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 11 — Feature map sizes diagram (SVG).

Output: figures/ch11_feature_shapes.svg
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from code.figures._save import save_png_pdf
plt.style.use('seaborn-v0_8')

def box(ax, x, y, w, h, text):
    r = plt.Rectangle((x, y), w, h, fc="#eef4ff", ec="#446")
    ax.add_patch(r)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center")

def main() -> None:
    out = Path("figures/ch11_feature_shapes.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.axis("off")
    # Set explicit limits so shapes are within view (avoid default 0..1 clipping)
    ax.set_xlim(0, 8.0)
    ax.set_ylim(0.0, 2.0)
    y = 0.6
    box(ax, 0.5, y, 1.4, 0.8, "28×28×1")
    box(ax, 2.4, y, 1.4, 0.8, "14×14×8")
    box(ax, 4.3, y, 1.4, 0.8, "7×7×16")
    box(ax, 6.2, y, 1.4, 0.8, "flatten 16×7×7")
    ax.annotate(
        "",
        xy=(2.4, y + 0.4),
        xytext=(1.9, y + 0.4),
        arrowprops=dict(arrowstyle="->"),
    )
    ax.annotate(
        "",
        xy=(4.3, y + 0.4),
        xytext=(3.8, y + 0.4),
        arrowprops=dict(arrowstyle="->"),
    )
    ax.annotate(
        "",
        xy=(6.2, y + 0.4),
        xytext=(5.7, y + 0.4),
        arrowprops=dict(arrowstyle="->"),
    )
    fig.tight_layout()
    fig.savefig(out, format="svg")
    print(f"Wrote {out}")
    save_png_pdf(out)
if __name__ == "__main__":
    main()
