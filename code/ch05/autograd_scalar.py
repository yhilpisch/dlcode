"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 5 — Autograd on a scalar: gradient of a simple function.

Run:
  python code/ch05/autograd_scalar.py
"""
from __future__ import annotations


def main() -> None:
    try:
        import torch  # type: ignore
    except Exception:
        print("PyTorch not installed.")
        return

    w = torch.tensor(5.0, requires_grad=True)
    f = (w - 2) ** 2
    f.backward()
    print("w=", float(w), "f=", float(f), "grad=", float(w.grad))

    # One gradient descent step
    with torch.no_grad():
        w -= 0.1 * w.grad
        w.grad.zero_()
    print("w after step=", float(w))


if __name__ == "__main__":
    main()

