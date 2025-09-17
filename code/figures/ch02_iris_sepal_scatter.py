"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Generate SVG figure for Chapter 2: Iris sepal length/width scatter.

Output:
  figures/iris_sepal_scatter.svg
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import datasets

plt.style.use('seaborn-v0_8')


def main() -> None:
    out = Path("figures/ch02_iris_sepal_scatter.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    iris = datasets.load_iris()
    X = iris.data[:, [0, 1]]  # sepal length, sepal width
    y = iris.target

    plt.figure(figsize=(4.2, 3.2))
    for cls, marker, label in [
        (0, "o", iris.target_names[0]),
        (1, "s", iris.target_names[1]),
        (2, "^", iris.target_names[2]),
    ]:
        idx = y == cls
        plt.scatter(X[idx, 0], X[idx, 1], marker=marker, label=label, s=28)
    plt.xlabel("sepal length (cm)")
    plt.ylabel("sepal width (cm)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out, format="svg")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
