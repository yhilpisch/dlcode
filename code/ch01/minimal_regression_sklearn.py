"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 1 — Minimal Linear Regression with scikit-learn

Run:
  python code/ch01/minimal_regression_sklearn.py
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def main() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0.0, 1.0, 2.1, 2.9], dtype=float)
    model = LinearRegression().fit(X, y)
    print("coef=", model.coef_.ravel().tolist(), "intercept=", float(model.intercept_))

    pred = model.predict(X)
    for xi, yi, pi in zip(X.ravel(), y, pred):
        print(f"x={xi:.1f}  y={yi:.2f}  pred={pi:.2f}")


if __name__ == "__main__":
    main()

