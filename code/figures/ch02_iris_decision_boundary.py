"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Generate SVG decision boundary for Chapter 2 (Iris, petal features).

Output:
  figures/ch02_iris_decision_boundary.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from code.figures._save import save_png_pdf

plt.style.use('seaborn-v0_8')


def main() -> None:
    out = Path("figures/ch02_iris_decision_boundary.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(X, y)

    xmin, xmax = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    ymin, ymax = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 300),
        np.linspace(ymin, ymax, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = pipe.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(4.2, 3.2))
    plt.contourf(
        xx,
        yy,
        zz,
        alpha=0.25,
        levels=[-0.5, 0.5, 1.5, 2.5],
        cmap="coolwarm",
    )
    for cls, marker, label in [
        (0, "o", iris.target_names[0]),
        (1, "s", iris.target_names[1]),
        (2, "^", iris.target_names[2]),
    ]:
        idx = y == cls
        plt.scatter(X[idx, 0], X[idx, 1], marker=marker, label=label, s=25)
    plt.xlabel("petal length (cm)")
    plt.ylabel("petal width (cm)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out, format="svg")
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
