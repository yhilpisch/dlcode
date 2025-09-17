"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 9 — Train TinyMLP with DataLoader on moons.

Run:
  python code/ch09/dataloader_moons_train.py
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_dl(model: nn.Module, loader: DataLoader, *, epochs: int = 10, lr: float = 5e-3) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for Xb, yb in loader:
            loss = loss_fn(model(Xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


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
    model = train_dl(model, train_loader, epochs=10)
    model.eval()
    with torch.no_grad():
        acc = ((model(X_te).argmax(dim=1) == y_te).float().mean().item())
    print("test_acc=", round(acc, 3))


if __name__ == "__main__":
    main()
