"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

"""
from pathlib import Path
import matplotlib.pyplot as plt


def main() -> None:
    out = Path("figures/ch15_throughput_scaling.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    gpus = [1, 2, 4, 8]
    thr = [1.0, 1.9, 3.6, 6.2]  # relative samples/sec (illustrative)
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    ax.bar([str(x) for x in gpus], thr, color="#4daf4a")
    ax.set_xlabel("# GPUs")
    ax.set_ylabel("relative throughput")
    ax.set_ylim(0, max(thr) * 1.2)
    for i, v in enumerate(thr):
        ax.text(i, v + 0.05, f"{v:.1f}×", ha="center", fontsize=9)
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")


if __name__ == "__main__":
    main()

