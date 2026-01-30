"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

"""
from pathlib import Path
import matplotlib.pyplot as plt
from code.figures._save import save_png_pdf


def main() -> None:
    out = Path("figures/ch15_zero_sharding.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.axis("off")

    # Draw 3 GPUs and shard parameter/optimizer states
    labels = ["params shard", "optimizer shard", "grads shard"]
    colors = ["#fee8c8", "#c7e9c0", "#c6dbef"]
    for i in range(3):
        x = 0.1 + i * 0.28
        ax.add_patch(
            plt.Rectangle((x, 0.30), 0.22, 0.52, ec="#444", fc="#f7f7f7")
        )
        # Move GPU label well above the box to avoid overlap with title
        ax.text(x + 0.11, 0.86, f"GPU{i}", ha="center", va="bottom", fontsize=10)
        base_y = 0.34  # ensure inner boxes are fully inside the outer frame
        for j, (lab, fc) in enumerate(zip(labels, colors)):
            y0 = base_y + j * 0.15
            ax.add_patch(
                plt.Rectangle((x + 0.02, y0), 0.18, 0.12, ec="#666", fc=fc)
            )
            ax.text(
                x + 0.11, y0 + 0.06, lab, ha="center", va="center", fontsize=8
            )

    # Lift the title further to prevent collisions with GPU labels
    ax.text(
        0.5,
        0.94,
        "ZeRO: shard params/opt/grads across devices; all-gather on demand",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    save_png_pdf(out)


if __name__ == "__main__":
    main()
