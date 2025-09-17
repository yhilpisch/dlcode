"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 7 — Train/validation curves for tiny MLP on moons (SVG).

Output: figures/ch07_train_val_curves.svg
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


def main() -> None:
    out = Path("figures/ch07_train_val_curves.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Parameters
    W1 = torch.randn(2, 16, requires_grad=True)
    b1 = torch.zeros(16, requires_grad=True)
    W2 = torch.randn(16, 2, requires_grad=True)
    b2 = torch.zeros(2, requires_grad=True)
    with torch.no_grad():
        W1.mul_(0.5); W2.mul_(0.5)
    params = [W1, b1, W2, b2]

    def forward(X: torch.Tensor) -> torch.Tensor:
        h = torch.relu(X @ W1 + b1)
        return h @ W2 + b2

    def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(logits, y)

    def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
        return (logits.argmax(dim=1) == y).float().mean().item()

    epochs = 60
    train_loss, val_loss, val_acc = [], [], []
    for _ in range(epochs):
        logits = forward(X_tr)
        loss = loss_fn(logits, y_tr)
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()
        with torch.no_grad():
            for p in params:
                p -= 0.1 * p.grad
        train_loss.append(float(loss.detach()))
        with torch.no_grad():
            val_logits = forward(X_val)
            val_loss.append(float(loss_fn(val_logits, y_val).detach()))
            val_acc.append(accuracy(val_logits, y_val))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    e = range(1, epochs+1)
    axes[0].plot(e, train_loss, marker='o')
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('train loss'); axes[0].set_title('Train loss')
    axes[1].plot(e, val_loss, marker='o', color='C1')
    axes[1].set_xlabel('epoch'); axes[1].set_ylabel('val loss'); axes[1].set_title('Validation loss')
    axes[2].plot(e, val_acc, marker='o', color='C2')
    axes[2].set_xlabel('epoch'); axes[2].set_ylabel('val accuracy'); axes[2].set_title('Validation accuracy')
    fig.tight_layout(); fig.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
