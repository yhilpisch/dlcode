"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 9 — Effect of batch size on loss curves (SVG).

Output: figures/ch09_batchsize_loss.svg
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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from code.figures._save import save_png_pdf


def train_loss_curve(batch_size: int, epochs: int = 15) -> list[float]:
    torch.manual_seed(0)
    X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
    )
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for Xb, yb in loader:
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            n += 1
        losses.append(tot / max(n, 1))
    return losses


def main() -> None:
    out = Path("figures/ch09_batchsize_loss.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    sizes = [16, 64, 256]
    curves = {bs: train_loss_curve(bs) for bs in sizes}
    e = range(1, len(next(iter(curves.values()))) + 1)
    plt.figure(figsize=(6.8, 3.2))
    for bs, ls in curves.items():
        plt.plot(e, ls, marker="o", label=f"batch={bs}")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Batch size vs loss curve")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out, format="svg")
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
