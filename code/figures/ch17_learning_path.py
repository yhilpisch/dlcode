"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

"""
from pathlib import Path
import matplotlib.pyplot as plt


def main() -> None:
    out = Path("figures/ch17_learning_path.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Base path boxes
    box_w, box_h = 0.16, 0.22
    base = [
        ("Foundations", 0.08, 0.60),
        ("Projects", 0.30, 0.60),
        ("Specialize", 0.52, 0.60),
    ]
    for label, x, y in base:
        ax.add_patch(plt.Rectangle((x, y), box_w, box_h, ec="#444", fc="#eef5ff"))
        ax.text(x + box_w/2, y + box_h/2, label, ha="center", va="center", fontsize=10)

    # Arrows between base boxes
    ax.annotate("", xy=(0.30, 0.71), xytext=(0.24, 0.71), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(0.52, 0.71), xytext=(0.46, 0.71), arrowprops=dict(arrowstyle="->", lw=1.2))

    # Specializations container
    cont_x, cont_y, cont_w, cont_h = 0.72, 0.48, 0.22, 0.40
    ax.add_patch(plt.Rectangle((cont_x, cont_y), cont_w, cont_h, ec="#444", fc="#ffffff"))
    ax.text(cont_x + cont_w/2, cont_y + cont_h + 0.03, "Specializations", ha="center", va="bottom", fontsize=10)

    # Rows inside container (five stacked rows)
    rows = ["CV", "NLP/LLMs", "Generative", "RL", "Systems"]
    row_h = cont_h / len(rows)
    for i, label in enumerate(rows):
        y = cont_y + cont_h - (i + 1) * row_h
        ax.add_patch(plt.Rectangle((cont_x, y), cont_w, row_h, ec="#ddd", fc="#f8fbff"))
        ax.text(cont_x + 0.06, y + row_h / 2, label, ha="left", va="center", fontsize=10)

    # Fan-out arrows to each row (to left edge of each row) from Specialize
    entry = (cont_x + 0.01, cont_y + cont_h / 2)
    for i in range(len(rows)):
        y_mid = cont_y + cont_h - (i + 0.5) * row_h
        ax.annotate(
            "",
            xy=(cont_x + 0.03, y_mid),
            xytext=(0.67, 0.71),
            arrowprops=dict(arrowstyle="->", lw=1.0, color="#1f77b4"),
        )

    fig.savefig(out, bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":
    main()
