"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 4 — Learning curve (train/test accuracy vs training size) for Logistic
Regression on moons (SVG).

Output: figures/ch04_learning_curve_logit.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from code.figures._save import save_png_pdf


def main() -> None:
    out = Path("figures/ch04_learning_curve_logit.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    X, y = make_moons(n_samples=2000, noise=0.25, random_state=0)
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    fracs = np.linspace(0.1, 1.0, 10)
    train_scores, test_scores, sizes = [], [], []
    for f in fracs:
        n = max(20, int(len(X_tr_full) * f))
        X_tr, y_tr = X_tr_full[:n], y_tr_full[:n]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X_tr, y_tr)
        train_scores.append(clf.score(X_tr, y_tr))
        test_scores.append(clf.score(X_te, y_te))
        sizes.append(n)

    plt.figure(figsize=(5.2, 3.4))
    plt.plot(sizes, train_scores, label='train', marker='o')
    plt.plot(sizes, test_scores, label='test', marker='o')
    plt.xlabel("training set size")
    plt.ylabel("accuracy")
    plt.title('Learning curve (logistic regression on moons)')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out, format="svg")
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
