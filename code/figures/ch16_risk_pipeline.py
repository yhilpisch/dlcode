"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

"""
from pathlib import Path
import matplotlib.pyplot as plt
from code.figures._save import save_png_pdf


def main() -> None:
    out = Path("figures/ch16_risk_pipeline.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 2.8))
    ax.axis("off")

    stages = ["Data", "Labeling", "Training", "Evaluation", "Deployment"]
    x0 = 0.06
    box_w, box_h = 0.16, 0.22
    y_box = 0.46
    y_arrow = 0.78
    for i, s in enumerate(stages):
        x = x0 + i * 0.19
        rect = plt.Rectangle((x, y_box), box_w, box_h, ec="#444", fc="#eef5ff")
        ax.add_patch(rect)
        ax.text(
            x + box_w / 2,
            y_box + box_h / 2,
            s,
            ha="center",
            va="center",
            fontsize=10,
        )
        if i < len(stages) - 1:
            x_left = x + box_w + 0.01
            x_right = x + box_w + 0.08
            ax.annotate(
                "",
                xy=(x_right, y_arrow),
                xytext=(x_left, y_arrow),
                arrowprops=dict(arrowstyle="<->", lw=1.2, color="#1f77b4"),
            )

    # Callouts (aligned under each box)
    callouts = [
        "provenance, consent, bias",
        "guidelines, QA, pay",
        "privacy, robustness",
        "fairness metrics",
        "guardrails, feedback",
    ]
    for i, text in enumerate(callouts):
        x = x0 + i * 0.19 + box_w / 2
        ax.text(x, 0.24, text, ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    save_png_pdf(out)


if __name__ == "__main__":
    main()
