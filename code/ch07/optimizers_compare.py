"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 7 — Compare SGD and Adam on moons.

Run:
  python code/ch07/optimizers_compare.py
"""
from __future__ import annotations

import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def run(opt_name: str, lr: float, steps: int = 400) -> float:
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
    opt = torch.optim.SGD(params, lr=lr) if opt_name == "sgd" else torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        h = torch.relu(X_tr @ W1 + b1)
        logits = h @ W2 + b2
        loss = torch.nn.functional.cross_entropy(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = ( (torch.relu(X_te @ W1 + b1) @ W2 + b2).argmax(dim=1) == y_te).float().mean().item()
    return acc


def main() -> None:
    torch.manual_seed(0)
    print("sgd=", round(run("sgd", 0.1), 3))
    print("adam=", round(run("adam", 0.01), 3))


if __name__ == "__main__":
    main()

