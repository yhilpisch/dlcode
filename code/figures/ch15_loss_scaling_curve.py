"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    out = Path("figures/ch15_loss_scaling_curve.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    steps = np.arange(0, 200)
    scale = np.clip(16384 / (1 + 0.01 * steps), 128, None)
    overflows = (steps % 37 == 0) & (steps > 0)
    for i, of in enumerate(overflows):
        if of:
            scale[i:] = scale[i:] / 2

    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    ax.plot(steps, scale, color="#1f77b4", lw=1.8)
    ax.scatter(steps[overflows], scale[overflows], color="#d62728", s=18, zorder=3, label="overflow")
    ax.set_xlabel("step")
    ax.set_ylabel("loss scale")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")


if __name__ == "__main__":
    main()

