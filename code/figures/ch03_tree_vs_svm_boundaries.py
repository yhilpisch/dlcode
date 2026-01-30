"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 3 — Decision boundaries on moons: Decision Tree vs RBF SVM (SVG).

Output: figures/ch03_tree_vs_svm_boundaries.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from code.figures._save import save_png_pdf
plt.style.use('seaborn-v0_8')


def plot_boundary(ax, clf, X, y, title: str) -> None:
    xmin, xmax = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    ymin, ymax = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 300), np.linspace(ymin, ymax, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = clf.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.25, levels=[-0.5, 0.5, 1.5], cmap="coolwarm")
    for cls, marker, label in [(0, "o", "class 0"), (1, "^", "class 1")]:
        idx = y == cls
        ax.scatter(X[idx, 0], X[idx, 1], marker=marker, s=18, alpha=0.9, label=label)
    ax.set_title(title)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")


def main() -> None:
    out = Path("figures/ch03_tree_vs_svm_boundaries.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    X, y = make_moons(n_samples=300, noise=0.25, random_state=0)

    tree = DecisionTreeClassifier(max_depth=4, random_state=0)
    tree.fit(X, y)
    rbf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
    rbf.fit(X, y)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharex=True, sharey=True)
    plot_boundary(axes[0], tree, X, y, "Decision Tree (depth=4)")
    plot_boundary(axes[1], rbf, X, y, "RBF SVM")
    for ax in axes:
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(out, format="svg")
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

