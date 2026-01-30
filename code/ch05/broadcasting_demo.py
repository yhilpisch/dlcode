"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 5 — Broadcasting demo.

Run:
  python code/ch05/broadcasting_demo.py
"""
from __future__ import annotations


def main() -> None:
    try:
        import torch  # type: ignore
    except Exception:
        print("PyTorch not installed.")
        return

    a = torch.arange(3, dtype=torch.float32).reshape(3, 1)
    b = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    c = a + b
    print("a shape:", tuple(a.shape), "\n", a)
    print("b shape:", tuple(b.shape), "\n", b)
    print("c=a+b shape:", tuple(c.shape), "\n", c)


if __name__ == "__main__":
    main()

