"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Generate SVG confusion matrix for Chapter 2 (Iris, petal features).

Output:
  figures/ch02_iris_confusion_matrix.svg
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from code.figures._save import save_png_pdf

plt.style.use('seaborn-v0_8')


def main() -> None:
    out = Path("figures/ch02_iris_confusion_matrix.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title("Iris — Logistic Regression (petal features)")
    ax.set_xticks([0, 1, 2], labels=iris.target_names, rotation=45, ha='right')
    ax.set_yticks([0, 1, 2], labels=iris.target_names)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.savefig(out, format="svg")
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
