"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 10 — Decision boundary with/without weight decay (SVG).

Output: figures/ch10_boundary_l2.svg
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
from code.figures._save import save_png_pdf


def train_model(weight_decay: float):
    torch.manual_seed(0)
    X, y = make_moons(n_samples=800, noise=0.35, random_state=0)
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 2))
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(100):
        model.train()
        logits = model(X_tr)
        loss = loss_fn(logits, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, X, y


def main() -> None:
    out = Path("figures/ch10_boundary_l2.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    m0, X, y = train_model(0.0)
    m1, _, _ = train_model(5e-2)
    xmin, xmax = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    ymin, ymax = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 250),
        np.linspace(ymin, ymax, 250),
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        zz0 = m0(grid).argmax(1).numpy().reshape(xx.shape)
        zz1 = m1(grid).argmax(1).numpy().reshape(xx.shape)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharex=True, sharey=True)
    for ax, zz, title in zip(
        axes, [zz0, zz1], ["no L2", "weight decay 1e-2"]
    ):
        ax.contourf(
            xx,
            yy,
            zz,
            levels=[-0.5, 0.5, 1.5],
            cmap="coolwarm",
            alpha=0.25,
        )
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=10, label="0")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=10, label="1")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    save_png_pdf(out)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
