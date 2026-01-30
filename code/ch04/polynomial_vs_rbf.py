"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 4 — Polynomial logistic regression vs RBF SVC on two moons.

Run:
  python code/ch04/polynomial_vs_rbf.py
"""
from __future__ import annotations

from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC


def main() -> None:
    X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    poly_logit = make_pipeline(
        PolynomialFeatures(degree=3, include_bias=False),
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    ).fit(X_tr, y_tr)

    rbf_svc = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale"),
    ).fit(X_tr, y_tr)

    print(
        {
            "poly_logit": round(accuracy_score(y_te, poly_logit.predict(X_te)), 3),
            "rbf_svc": round(accuracy_score(y_te, rbf_svc.predict(X_te)), 3),
        }
    )


if __name__ == "__main__":
    main()
