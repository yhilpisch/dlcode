"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 2 — Scatter plot of two Iris features, colored by class.

Run:
  python code/ch02/scatter_features.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
from sklearn import datasets


def main() -> None:
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]  # petal length, petal width
    y = iris.target

    plt.figure(figsize=(4, 3))
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
