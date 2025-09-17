"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 8 — Minimal DataLoader demo with TinyMLP.

Run:
  python code/ch08/dataloader_demo.py
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 16, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_dl(model: nn.Module, loader: DataLoader, *, epochs: int = 10, lr: float = 5e-3, device: str = "cpu") -> nn.Module:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def evaluate_dl(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    model.eval(); model.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / max(total, 1)


def main() -> None:
    torch.manual_seed(0)
    X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    X_te = torch.tensor(X_te, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    y_te = torch.tensor(y_te, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=256)

    model = TinyMLP()
    model = train_dl(model, train_loader, epochs=10, lr=5e-3)
    acc = evaluate_dl(model, test_loader)
    print("test_acc=", round(acc, 3))


if __name__ == "__main__":
    main()

