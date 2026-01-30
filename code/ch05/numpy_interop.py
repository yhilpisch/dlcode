"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 5 — NumPy interop: sharing memory between torch and numpy.

Run:
  python code/ch05/numpy_interop.py
"""
from __future__ import annotations


def main() -> None:
    try:
        import torch  # type: ignore
        import numpy as np
    except Exception:
        print("PyTorch/NumPy not installed.")
        return

    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = torch.from_numpy(a)
    print("t before=\n", t)
    a *= 10
    print("t after np edit=\n", t)  # reflects change

    u = t.numpy()
    t += 1
    print("u after torch edit=\n", u)


if __name__ == "__main__":
    main()

