"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Quick environment check for Chapter 1.

Run:
  python -m code.env_check
"""
from __future__ import annotations

import platform


def main() -> None:
    print(f"Python: {platform.python_version()} ({platform.python_implementation()})")
    # Optional: report PyTorch if available
    try:
        import torch  # type: ignore

        device = (
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        print(f"PyTorch: {torch.__version__}  device: {device}")
    except Exception:
        print("PyTorch: not installed — using CPU-only examples in this chapter")


if __name__ == "__main__":
    main()

