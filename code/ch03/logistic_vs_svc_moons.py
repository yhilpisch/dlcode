"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 3 — Logistic regression vs RBF SVC on moons.

Run:
  python code/ch03/logistic_vs_svc_moons.py
"""
from __future__ import annotations

from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main() -> None:
    X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    logit = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, y_tr)
    rbf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale"),
    ).fit(X_tr, y_tr)

    print("Logistic acc=", round(accuracy_score(y_te, logit.predict(X_te)), 3))
    print("RBF SVC acc=", round(accuracy_score(y_te, rbf.predict(X_te)), 3))


if __name__ == "__main__":
    main()
