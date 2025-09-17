"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 10 — Validation curves with/without dropout (SVG).

Output: figures/ch10_curves_dropout.svg
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


def curves(p: float) -> list[float]:
    torch.manual_seed(0)
    X, y = make_moons(n_samples=1000, noise=0.40, random_state=0)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.7, random_state=42, stratify=y)
    X_va, _, y_va, _ = train_test_split(X_tmp, y_tmp, test_size=0.7, random_state=42, stratify=y_tmp)
    X_tr = torch.tensor(X_tr, dtype=torch.float32); y_tr = torch.tensor(y_tr, dtype=torch.long)
    X_va = torch.tensor(X_va, dtype=torch.float32); y_va = torch.tensor(y_va, dtype=torch.long)
    model = nn.Sequential(nn.Linear(2,256), nn.ReLU(), nn.Dropout(p), nn.Linear(256,2))
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()
    va: list[float] = []
    for _ in range(150):
        model.train(); logits = model(X_tr); loss = loss_fn(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval();
        with torch.no_grad(): va.append(float(loss_fn(model(X_va), y_va)))
    return va


def main() -> None:
    out = Path('figures/ch10_curves_dropout.svg'); out.parent.mkdir(parents=True, exist_ok=True)
    v0 = curves(0.0); v1 = curves(0.6)
    e = range(1, len(v0)+1)
    plt.figure(figsize=(5.6,3.0))
    plt.plot(e, v0, label='dropout p=0.0')
    plt.plot(e, v1, label='dropout p=0.3')
    plt.xlabel('epoch'); plt.ylabel('val loss'); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out, format='svg')
    print(f"Wrote {out}")

if __name__ == '__main__':
    main()
