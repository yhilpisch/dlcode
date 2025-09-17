"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 5 — Tensors basics: creation, shapes, dtypes, devices.

Run:
  python code/ch05/tensors_basics.py
"""
from __future__ import annotations

def main() -> None:
    try:
        import torch  # type: ignore
    except Exception as e:
        print("PyTorch not installed. Install torch to run Chapter 5 demos.")
        return

    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print("x=\n", x)
    print("shape=", tuple(x.shape), "dtype=", x.dtype, "device=", x.device)
    print("x.T=\n", x.T)
    print("x.mean(dim=0)=", x.mean(dim=0))
    print("x.mean(dim=1)=", x.mean(dim=1))

    # Device move if CUDA is available
    if torch.cuda.is_available():
        x_cuda = x.to('cuda')
        print("moved to:", x_cuda.device)


if __name__ == "__main__":
    main()

