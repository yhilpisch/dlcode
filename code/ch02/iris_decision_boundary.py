"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 2 — Decision boundary plot for Iris using two features.

Run:
  python code/ch02/iris_decision_boundary.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]  # petal length, petal width
    y = iris.target

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(X, y)

    xmin, xmax = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    ymin, ymax = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = pipe.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(4, 3))
    plt.contourf(xx, yy, zz, alpha=0.2, levels=[-0.5, 0.5, 1.5, 2.5], cmap="coolwarm")
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
    plt.show()


if __name__ == "__main__":
    main()
