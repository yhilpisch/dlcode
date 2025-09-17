"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 9 — Standardization scatter (raw vs standardized) (SVG).

Output: figures/ch09_standardize_scatter.svg
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


def main() -> None:
    out = Path('figures/ch09_standardize_scatter.svg')
    out.parent.mkdir(parents=True, exist_ok=True)
    X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    xmin, xmax = X[:,0].min()-0.5, X[:,0].max()+0.5
    ymin, ymax = X[:,1].min()-0.5, X[:,1].max()+0.5
    fig, axes = plt.subplots(1,2, figsize=(7.6,3.2), sharex=False, sharey=False)
    for ax, Z, title in zip(axes, [X, Xs], ['raw', 'standardized']):
        ax.scatter(Z[y==0,0], Z[y==0,1], s=10, label='0')
        ax.scatter(Z[y==1,0], Z[y==1,1], s=10, label='1')
        ax.set_title(title)
        if title=='raw':
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, format='svg')
    print(f"Wrote {out}")

if __name__=='__main__':
    main()
