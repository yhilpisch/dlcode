"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 3 — Probability contour for logistic regression on moons (SVG).

Output: figures/ch03_logit_prob_contour.svg
"""
from __future__ import annotations

from pathlib import Path
import os
import numpy as np

# Ensure Matplotlib can write caches in sandboxed environments
cache_dir = Path('.cache/matplotlib')
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(cache_dir.resolve()))
os.environ.setdefault('XDG_CACHE_HOME', str(Path('.cache').resolve()))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def main() -> None:
    out = Path("figures/ch03_logit_prob_contour.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
    clf_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf_pipe.fit(X, y)

    xmin, xmax = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    ymin, ymax = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
    # Slightly lower grid resolution to be lighter in restricted sandboxes
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # Compute probabilities manually to avoid heavy predict_proba calls in sandboxed envs
    scaler = clf_pipe.named_steps['standardscaler']
    lr = clf_pipe.named_steps['logisticregression']
    grid_std = (grid - scaler.mean_) / scaler.scale_
    logits = grid_std @ lr.coef_.ravel() + lr.intercept_[0]
    proba = (1.0 / (1.0 + np.exp(-logits))).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    cs = ax.contourf(xx, yy, proba, levels=21, cmap='coolwarm', alpha=0.8)
    ax.contour(xx, yy, proba, levels=[0.5], colors='k', linewidths=1.2)
    for cls, marker, label in [(0, 'o', 'class 0'), (1, '^', 'class 1')]:
        idx = y == cls
        ax.scatter(X[idx, 0], X[idx, 1], marker=marker, s=18, alpha=0.9, label=label)
    fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04, label='P(class=1)')
    ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.legend(frameon=False, loc='upper right')
    fig.tight_layout(); fig.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
