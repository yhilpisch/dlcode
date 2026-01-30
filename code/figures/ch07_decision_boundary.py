"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 7 — Decision boundary for trained tiny MLP on moons (SVG).

Output: figures/ch07_decision_boundary.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from code.figures._save import save_png_pdf


def main() -> None:
    out = Path("figures/ch07_decision_boundary.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)

    W1 = torch.randn(2, 16, requires_grad=True)
    b1 = torch.zeros(16, requires_grad=True)
    W2 = torch.randn(16, 2, requires_grad=True)
    b2 = torch.zeros(2, requires_grad=True)
    with torch.no_grad():
        W1.mul_(0.5); W2.mul_(0.5)
    params = [W1, b1, W2, b2]
    for _ in range(400):
        h = torch.relu(X_tr @ W1 + b1)
        logits = h @ W2 + b2
        loss = torch.nn.functional.cross_entropy(logits, y_tr)
        for p in params:
            if p.grad is not None: p.grad.zero_()
        loss.backward()
        with torch.no_grad():
            for p in params: p -= 0.1 * p.grad

    xmin, xmax = X[:,0].min()-0.4, X[:,0].max()+0.4
    ymin, ymax = X[:,1].min()-0.4, X[:,1].max()+0.4
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 250), np.linspace(ymin, ymax, 250))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        h = torch.relu(grid @ W1 + b1)
        zz = (h @ W2 + b2).argmax(dim=1).numpy().reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.contourf(xx, yy, zz, levels=[-0.5,0.5,1.5], cmap='coolwarm', alpha=0.25)
    ax.scatter(X[y==0,0], X[y==0,1], s=10, label='0')
    ax.scatter(X[y==1,0], X[y==1,1], s=10, label='1')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, format='svg')
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
