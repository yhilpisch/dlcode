"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 6 — XOR decision boundary: linear vs 2-layer MLP (SVG).

Output: figures/ch06_xor_boundary.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')


def make_xor(n=200, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    X = X + rng.normal(0, noise, size=X.shape)
    return X, y


def train_mlp(X, y, hidden=8, lr=0.1, steps=2000):
    import torch
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    W1 = torch.randn(2, hidden, requires_grad=True)
    b1 = torch.zeros(hidden, requires_grad=True)
    W2 = torch.randn(hidden, 2, requires_grad=True)
    b2 = torch.zeros(2, requires_grad=True)
    with torch.no_grad():
        W1.mul_(0.5); W2.mul_(0.5)
    params = [W1, b1, W2, b2]
    for _ in range(steps):
        h = torch.relu(X_t @ W1 + b1)
        logits = h @ W2 + b2
        loss = torch.nn.functional.cross_entropy(logits, y_t)
        for p in params:
            if p.grad is not None: p.grad.zero_()
        loss.backward()
        with torch.no_grad():
            for p in params: p -= lr * p.grad
    return [p.detach().clone() for p in params]


def predict_mlp(X, W1, b1, W2, b2):
    import torch
    X_t = torch.tensor(X, dtype=torch.float32)
    h = torch.relu(X_t @ W1 + b1)
    logits = h @ W2 + b2
    return logits.argmax(dim=1).numpy()


def main() -> None:
    out = Path("figures/ch06_xor_boundary.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    X, y = make_xor(n=600, noise=0.15, seed=0)

    # Linear baseline
    from sklearn.linear_model import LogisticRegression
    lin = LogisticRegression().fit(X, y)

    # Train small MLP manually
    W1, b1, W2, b2 = train_mlp(X, y, hidden=8, lr=0.1, steps=1500)

    # Grid for decision regions
    xmin, xmax = X[:,0].min()-0.4, X[:,0].max()+0.4
    ymin, ymax = X[:,1].min()-0.4, X[:,1].max()+0.4
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 250), np.linspace(ymin, ymax, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz_lin = lin.predict(grid).reshape(xx.shape)
    zz_mlp = predict_mlp(grid, W1, b1, W2, b2).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True, sharey=True)
    for ax, zz, title in zip(axes, [zz_lin, zz_mlp], ['Linear', '2-layer MLP (ReLU)']):
        ax.contourf(xx, yy, zz, levels=[-0.5,0.5,1.5], cmap='coolwarm', alpha=0.25)
        ax.scatter(X[y==0,0], X[y==0,1], s=10, label='0')
        ax.scatter(X[y==1,0], X[y==1,1], s=10, label='1')
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
