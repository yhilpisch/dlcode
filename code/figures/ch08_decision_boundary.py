"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 8 — Decision boundary for nn.Module MLP on moons (SVG).

Output: figures/ch08_decision_boundary.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def main() -> None:
    out = Path("figures/ch08_decision_boundary.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    model = nn.Sequential(nn.Linear(2,16), nn.ReLU(), nn.Linear(16,2))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(60):
        model.train(); opt.zero_grad()
        logits = model(X_tr)
        loss = loss_fn(logits, y_tr)
        loss.backward(); opt.step()

    xmin, xmax = X[:,0].min()-0.4, X[:,0].max()+0.4
    ymin, ymax = X[:,1].min()-0.4, X[:,1].max()+0.4
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 250), np.linspace(ymin, ymax, 250))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        zz = model(grid).argmax(dim=1).numpy().reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.contourf(xx, yy, zz, levels=[-0.5,0.5,1.5], cmap='coolwarm', alpha=0.25)
    ax.scatter(X[y==0,0], X[y==0,1], s=10, label='0')
    ax.scatter(X[y==1,0], X[y==1,1], s=10, label='1')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
