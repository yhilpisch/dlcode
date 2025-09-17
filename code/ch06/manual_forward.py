"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 6 — Manual forward pass of a 2-layer MLP with ReLU.

Run:
  python code/ch06/manual_forward.py
"""
from __future__ import annotations


def main() -> None:
    import torch
    torch.manual_seed(0)
    x = torch.randn(5, 2)
    W1 = torch.randn(2, 4)
    b1 = torch.randn(4)
    W2 = torch.randn(4, 1)
    b2 = torch.randn(1)
    h = torch.relu(x @ W1 + b1)
    y = h @ W2 + b2
    print("x shape:", tuple(x.shape), "y shape:", tuple(y.shape))


if __name__ == "__main__":
    main()

