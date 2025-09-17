"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 9 — Shuffle vs no-shuffle loss curves (SVG).

Output: figures/ch09_shuffle_loss.svg
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def run(shuffle: bool, epochs=12):
    torch.manual_seed(0)
    X,y = make_moons(n_samples=600, noise=0.25, random_state=0)
    X_tr, _, y_tr, _ = train_test_split(X,y,test_size=0.25, random_state=42, stratify=y)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    if shuffle:
        # Regular shuffled loader (recommended default)
        loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True)
    else:
        # Construct an adversarial, ordered loader (grouped by label) to illustrate why shuffling helps.
        idx = torch.argsort(y_tr_t)
        X_tr_ord, y_tr_ord = X_tr_t[idx], y_tr_t[idx]
        loader = DataLoader(TensorDataset(X_tr_ord, y_tr_ord), batch_size=64, shuffle=False)
    model = nn.Sequential(nn.Linear(2,16), nn.ReLU(), nn.Linear(16,2))
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn=nn.CrossEntropyLoss(); losses=[]
    for _ in range(epochs):
        model.train(); tot=n=0,0
        tot, n = 0.0, 0
        for Xb,yb in loader:
            loss=loss_fn(model(Xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); n += 1
        losses.append(tot/max(n,1))
    return losses

def main():
    out = Path('figures/ch09_shuffle_loss.svg'); out.parent.mkdir(parents=True, exist_ok=True)
    ls = run(True); ln = run(False)
    e=range(1,len(ls)+1)
    plt.figure(figsize=(5.8,3.0))
    plt.plot(e, ls, marker='o', label='shuffle=True')
    plt.plot(e, ln, marker='o', label='shuffle=False')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.title('Shuffle vs no-shuffle (loss)')
    plt.legend(frameon=False); plt.tight_layout(); plt.savefig(out, format='svg')
    print(f"Wrote {out}")

if __name__=='__main__':
    main()
