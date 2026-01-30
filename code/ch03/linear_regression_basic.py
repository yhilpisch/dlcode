"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 3 — Minimal linear regression with metrics.

Run:
  python code/ch03/linear_regression_basic.py
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main() -> None:
    rng = np.random.default_rng(0)
    X = np.linspace(0, 5, 30).reshape(-1, 1)
    y = 1.2 * X.ravel() + 0.5 + rng.normal(0, 0.35, size=X.shape[0])

    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    print(
        "coef=",
        model.coef_.ravel().tolist(),
        "intercept=",
        float(model.intercept_),
    )
    print(
        f"MAE={mean_absolute_error(y, pred):.3f} "
        f"MSE={mean_squared_error(y, pred):.3f} "
        f"R2={r2_score(y, pred):.3f}"
    )


if __name__ == "__main__":
    main()
