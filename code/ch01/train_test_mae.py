"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 1 — Train/Test Split and MAE Metric

Run:
  python code/ch01/train_test_mae.py
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def main() -> None:
    rng = np.random.default_rng(0)
    X = np.linspace(0, 4, 20, dtype=float).reshape(-1, 1)
    y = 0.95 * X.ravel() + 0.1 + rng.normal(0, 0.05, size=X.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE on test: {mae:.4f}")
    print("coef=", model.coef_.ravel().tolist(), "intercept=", float(model.intercept_))


if __name__ == "__main__":
    main()

