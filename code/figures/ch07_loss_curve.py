"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 7 — Training loss curve (SVG).

Output: figures/ch07_loss_curve.svg
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from code.figures._save import save_png_pdf


def main() -> None:
    out = Path("figures/ch07_loss_curve.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    W1 = torch.randn(2, 16, requires_grad=True)
    b1 = torch.zeros(16, requires_grad=True)
    W2 = torch.randn(16, 2, requires_grad=True)
    b2 = torch.zeros(2, requires_grad=True)
    with torch.no_grad():
        W1.mul_(0.5)
        W2.mul_(0.5)
    params = [W1, b1, W2, b2]
    lr, steps = 0.1, 400
    losses = []
    for t in range(steps):
        h = torch.relu(X_tr @ W1 + b1)
        logits = h @ W2 + b2
        loss = torch.nn.functional.cross_entropy(logits, y_tr)
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()
        with torch.no_grad():
            for p in params:
                p -= lr * p.grad
        if (t + 1) % 10 == 0:
            # Detach to avoid warning about converting a tensor requiring grad
            losses.append(loss.detach().item())
    xs = np.arange(10, steps + 1, 10)
    plt.figure(figsize=(5.0, 3.2))
    plt.plot(xs, losses, marker='o')
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(out, format="svg")
    save_png_pdf(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
