"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 4 — Model complexity curve for Decision Tree on moons (SVG).

Output: figures/ch04_tree_complexity_curve.svg
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
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    out = Path("figures/ch04_tree_complexity_curve.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    depths = list(range(1, 16))
    train_scores, test_scores = [], []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=0)
        clf.fit(X_tr, y_tr)
        train_scores.append(clf.score(X_tr, y_tr))
        test_scores.append(clf.score(X_te, y_te))

    plt.figure(figsize=(5.2, 3.4))
    plt.plot(depths, train_scores, label='train', marker='o')
    plt.plot(depths, test_scores, label='test', marker='o')
    plt.xlabel('max_depth'); plt.ylabel('accuracy')
    plt.title('Under/overfitting across model complexity (tree depth)')
    plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

