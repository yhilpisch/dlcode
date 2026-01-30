"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 3 — Decision Tree on moons dataset.

Run:
  python code/ch03/decision_tree_moons.py
"""
from __future__ import annotations

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr, y_tr)
    print("depth=", tree.get_depth(), "leaves=", tree.get_n_leaves())
    print(
        "train_acc=", round(tree.score(X_tr, y_tr), 3),
        "test_acc=", round(tree.score(X_te, y_te), 3)
    )


if __name__ == "__main__":
    main()

