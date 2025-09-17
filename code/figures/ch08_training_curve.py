"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 8 — Training loss by epoch for nn.Module MLP (SVG).

Output: figures/ch08_training_curve.svg
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def main() -> None:
    out = Path("figures/ch08_training_curve.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    model = nn.Sequential(nn.Linear(2,16), nn.ReLU(), nn.Linear(16,2))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for _ in range(50):
        model.train()
        logits = model(X_tr)
        loss = loss_fn(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    plt.figure(figsize=(5.0,3.2)); plt.plot(range(1,51), losses, marker='o')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.title('Training loss (nn.Module)')
    plt.tight_layout(); plt.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
