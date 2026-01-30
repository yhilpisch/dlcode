"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 7 — Train a tiny MLP with manual tensors (SGD step).

Run:
  python code/ch07/train_mlp_sgd.py
"""
from __future__ import annotations

import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def main() -> None:
    torch.manual_seed(0)
    X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    X_te = torch.tensor(X_te, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    y_te = torch.tensor(y_te, dtype=torch.long)

    W1 = torch.randn(2, 16, requires_grad=True)
    b1 = torch.zeros(16, requires_grad=True)
    W2 = torch.randn(16, 2, requires_grad=True)
    b2 = torch.zeros(2, requires_grad=True)
    params = [W1, b1, W2, b2]

    with torch.no_grad():
        W1.mul_(0.5); W2.mul_(0.5)

    def forward(X: torch.Tensor) -> torch.Tensor:
        h = torch.relu(X @ W1 + b1)
        return h @ W2 + b2

    def acc(logits: torch.Tensor, y: torch.Tensor) -> float:
        return (logits.argmax(dim=1) == y).float().mean().item()

    lr, steps = 0.1, 400
    for _ in range(steps):
        logits = forward(X_tr)
        loss = torch.nn.functional.cross_entropy(logits, y_tr)
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()
        with torch.no_grad():
            for p in params:
                p -= lr * p.grad
    print("test_acc=", round(acc(forward(X_te), y_te), 3))


if __name__ == "__main__":
    main()

