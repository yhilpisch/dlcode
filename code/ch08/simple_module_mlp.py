"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 8 — Tiny MLP as nn.Module with training/eval and checkpoint.

Run:
  python code/ch08/simple_module_mlp.py
"""
from __future__ import annotations

import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 16, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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

    model = TinyMLP()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
        return (logits.argmax(dim=1) == y).float().mean().item()

    for _ in range(50):
        model.train()
        logits = model(X_tr)
        loss = loss_fn(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        test_acc = accuracy(model(X_te), y_te)
    print("test_acc=", round(test_acc, 3))

    # Save and load
    path = 'tiny_mlp.pt'
    torch.save({'model': model.state_dict()}, path)
    loaded = TinyMLP(); loaded.load_state_dict(torch.load(path)['model'])
    loaded.eval()
    with torch.no_grad():
        print("loaded_acc=", round(accuracy(loaded(X_te), y_te), 3))


if __name__ == "__main__":
    main()
