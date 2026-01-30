"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 3 — Decision tree structure (hierarchical plot) on two moons.

Outputs:
  - figures/ch03_tree_structure.svg
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
from sklearn.tree import DecisionTreeClassifier, plot_tree
from code.figures._save import save_png_pdf


def main() -> None:
    out = Path("figures/ch03_tree_structure.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf.fit(X_tr, y_tr)

    fig = plt.figure(figsize=(9.0, 4.8))
    plot_tree(
        clf,
        feature_names=["x1", "x2"],
        class_names=["class 0", "class 1"],
        filled=True,
        impurity=True,
        rounded=True,
        fontsize=8,
    )
    plt.tight_layout(); fig.savefig(out, format="svg")
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

