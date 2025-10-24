#!/usr/bin/env python3
"""
Deep Learning with PyTorch — Chapter 6

Generate the XOR decision boundary figure contrasting a linear classifier
with a shallow ReLU MLP.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
FIG_PATH = FIG_DIR / "ch06_xor_boundary.png"


def make_xor_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return the canonical XOR dataset with two classes."""
    X = np.array(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 1, 1, 0], dtype=int)
    return X, y


def fit_models(X: np.ndarray, y: np.ndarray) -> tuple[LogisticRegression, MLPClassifier]:
    """Fit a linear classifier and a two-layer ReLU MLP to XOR."""
    linear = LogisticRegression(
        penalty="none",
        solver="lbfgs",
        max_iter=200,
        random_state=0,
    )
    linear.fit(X, y)

    mlp = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="relu",
        alpha=0.0,
        learning_rate_init=0.05,
        max_iter=2000,
        random_state=0,
    )
    mlp.fit(X, y)

    return linear, mlp


def decision_grid(
    model, xx: np.ndarray, yy: np.ndarray
) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Evaluate predictions for contour plotting."""
    mesh = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(mesh)
    return zz.reshape(xx.shape)


def plot_comparison(
    X: np.ndarray,
    y: np.ndarray,
    linear: LogisticRegression,
    mlp: MLPClassifier,
) -> None:
    """Create side-by-side decision boundary comparison plots."""
    xx, yy = np.meshgrid(
        np.linspace(-1.5, 1.5, 300),
        np.linspace(-1.5, 1.5, 300),
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharex=True, sharey=True)
    cmap = plt.cm.coolwarm

    for ax, model, title in zip(
        axes,
        (linear, mlp),
        ("Linear boundary", "Two-layer ReLU MLP"),
        strict=True,
    ):
        zz = decision_grid(model, xx, yy)
        ax.contourf(xx, yy, zz, levels=1, alpha=0.25, cmap=cmap)
        ax.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap=cmap,
            s=80,
            edgecolor="k",
        )
        ax.set_title(title)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_aspect("equal")

    fig.suptitle("XOR classification: linear vs two-layer MLP", y=1.05)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {FIG_PATH}")


def main() -> None:
    X, y = make_xor_dataset()
    linear, mlp = fit_models(X, y)
    plot_comparison(X, y, linear, mlp)


if __name__ == "__main__":
    main()
