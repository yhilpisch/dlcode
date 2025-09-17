"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 10 — Train/val curves with/without weight decay (SVG).

Output: figures/ch10_curves_l2.svg
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


def curves(weight_decay: float) -> tuple[list[float], list[float]]:
    torch.manual_seed(0)
    X, y = make_moons(n_samples=1000, noise=0.35, random_state=0)
    # smaller train set to make overfitting clearer
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.65, random_state=42, stratify=y
    )
    X_va, _, y_va, _ = train_test_split(X_tmp, y_tmp, test_size=0.65, random_state=42, stratify=y_tmp)
    X_tr = torch.tensor(X_tr, dtype=torch.float32); y_tr = torch.tensor(y_tr, dtype=torch.long)
    X_va = torch.tensor(X_va, dtype=torch.float32); y_va = torch.tensor(y_va, dtype=torch.long)
    model = nn.Sequential(nn.Linear(2,256), nn.ReLU(), nn.Linear(256,2))
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    tr: list[float] = []; va: list[float] = []
    for _ in range(150):
        model.train(); logits = model(X_tr); loss = loss_fn(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval();
        with torch.no_grad():
            tr.append(float(loss)); va.append(float(loss_fn(model(X_va), y_va)))
    return tr, va


def main() -> None:
    out = Path("figures/ch10_curves_l2.svg"); out.parent.mkdir(parents=True, exist_ok=True)
    tr0, va0 = curves(0.0)
    wd = 5e-3
    tr1, va1 = curves(wd)
    e = range(1, len(tr0)+1)
    fig, axes = plt.subplots(1,2, figsize=(7.6,3.0), sharey=True)
    axes[0].plot(e, tr0, label='train'); axes[0].plot(e, va0, label='val')
    axes[0].set_title('no weight decay'); axes[0].set_xlabel('epoch'); axes[0].set_ylabel('loss'); axes[0].legend(frameon=False)
    axes[1].plot(e, tr1, label='train'); axes[1].plot(e, va1, label='val')
    axes[1].set_title(f'weight decay {wd:g}'); axes[1].set_xlabel('epoch'); axes[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, format='svg')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
