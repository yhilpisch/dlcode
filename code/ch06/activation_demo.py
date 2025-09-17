"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 6 — Activation demo (sigmoid, tanh, ReLU outputs).

Run:
  python code/ch06/activation_demo.py
"""
from __future__ import annotations


def main() -> None:
    import torch
    x = torch.linspace(-4, 4, steps=9)
    print("sigmoid:", torch.sigmoid(x))
    print("tanh   :", torch.tanh(x))
    print("ReLU   :", torch.relu(x))


if __name__ == "__main__":
    main()

