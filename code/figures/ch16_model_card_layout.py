"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

"""
from pathlib import Path
import matplotlib.pyplot as plt


def main() -> None:
    out = Path("figures/ch16_model_card_layout.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.axis("off")

    sections = [
        ("Model Details", 0.80),
        ("Intended Use", 0.66),
        ("Data", 0.52),
        ("Metrics & Evaluation", 0.38),
        ("Limitations & Ethics", 0.24),
        ("Contact & Versioning", 0.10),
    ]
    for title, y in sections:
        rect = plt.Rectangle((0.08, y), 0.84, 0.12, ec="#444", fc="#f8fbff")
        ax.add_patch(rect)
        ax.text(0.50, y + 0.06, title, ha="center", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")


if __name__ == "__main__":
    main()

