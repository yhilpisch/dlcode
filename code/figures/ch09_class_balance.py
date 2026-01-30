"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 9 — Class imbalance and WeightedRandomSampler (SVG).

Output: figures/ch09_class_balance.svg
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.datasets import make_moons
from code.figures._save import save_png_pdf


def main() -> None:
    out = Path("figures/ch09_class_balance.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
    # create imbalance ~80/20 by masking class 1
    keep = (y == 0) | (
        (y == 1) & (np.random.default_rng(0).random(len(y)) < 0.25)
    )
    y_imb = y[keep]
    X_imb = X[keep]
    # counts before
    c0_before = int((y_imb == 0).sum())
    c1_before = int((y_imb == 1).sum())
    # weights inverse to class frequency
    class_count = np.bincount(y_imb)
    w = 1.0 / class_count
    sample_weights = w[y_imb]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y_imb),
        replacement=True,
    )
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_imb, dtype=torch.float32),
            torch.tensor(y_imb, dtype=torch.long),
        ),
        batch_size=64,
        sampler=sampler,
    )
    # sample one epoch and count
    c0_after = c1_after = 0
    for _, yb in loader:
        c0_after += int((yb == 0).sum())
        c1_after += int((yb == 1).sum())
    plt.figure(figsize=(6.8, 3.0))
    labels = ["class 0", "class 1"]
    before = [c0_before, c1_before]
    after = [c0_after, c1_after]
    x = np.arange(2)
    w = 0.35
    plt.bar(x - w / 2, before, width=w, label="original")
    plt.bar(x + w / 2, after, width=w, label="weighted sampler")
    plt.xticks(x, labels)
    plt.ylabel("count per epoch")
    plt.title("Class imbalance vs weighted sampling")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out, format="svg")
    save_png_pdf(out)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
