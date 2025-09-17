"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 2 — Iris classification with a scaler + logistic regression pipeline.

Run:
  python code/ch02/iris_logreg_pipeline.py
"""
from __future__ import annotations

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')


def main() -> None:
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]  # petal length, petal width
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
        )

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=iris.target_names)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
