"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 4 — Learning curve for logistic regression on moons.

Run:
  python code/ch04/logistic_learning_curve.py
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    X, y = make_moons(n_samples=2000, noise=0.25, random_state=0)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    train_sizes, train_scores, test_scores = learning_curve(
        pipeline,
        X,
        y,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=None,
    )

    print("size,train_acc,test_acc")
    for size, tr, te in zip(
        train_sizes,
        train_scores.mean(axis=1),
        test_scores.mean(axis=1),
    ):
        print(f"{int(size)},{tr:.3f},{te:.3f}")


if __name__ == "__main__":
    main()
