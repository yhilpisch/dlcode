"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 3 — Confusion matrix for logistic regression on moons (SVG).

Output: figures/ch03_logit_confusion.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from code.figures._save import save_png_pdf


def main() -> None:
    out = Path("figures/ch03_logit_confusion.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    clf = LogisticRegression().fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    im = ax.imshow(cm, interpolation='nearest', cmap='coolwarm')
    ax.set_title("Logistic regression — moons")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
