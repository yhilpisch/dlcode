"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 9 — DataLoader workers throughput (SVG).

Output: figures/ch09_workers_throughput.svg
"""
from __future__ import annotations
from pathlib import Path
import platform
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import torch
from torch.utils.data import Dataset, DataLoader
from code.figures._save import save_png_pdf

class SlowDataset(Dataset):
    def __init__(self, n: int = 400, delay: float = 0.001) -> None:
        self.n = n
        self.delay = delay

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i):
        time.sleep(self.delay)  # emulate small I/O/transform cost
        x = np.array([i % 10, (i * 7) % 10], dtype=np.float32)
        y = (x.sum() % 2).astype(np.int64)
        return torch.from_numpy(x), torch.tensor(y)

def measure(num_workers: int) -> float:
    ds = SlowDataset(n=300, delay=0.001)
    loader = DataLoader(
        ds, batch_size=64, shuffle=False, num_workers=num_workers
    )
    t0 = time.time()
    total = 0
    for _ in range(2):
        for xb, yb in loader:
            total += xb.size(0)
    dt = time.time() - t0
    return total / max(dt, 1e-6)

def main() -> None:
    # Ensure safe start method on macOS to avoid DataLoader hangs
    if platform.system() == "Darwin":
        try:
            torch.multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    out = Path("figures/ch09_workers_throughput.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = [0, 2, 4, 8]
    rates = [measure(w) for w in workers]
    plt.figure(figsize=(5.8, 3.0))
    plt.bar([str(w) for w in workers], rates)
    plt.xlabel("num_workers")
    plt.ylabel("samples/sec")
    plt.title("Loader throughput vs workers")
    plt.tight_layout()
    plt.savefig(out, format="svg")
    save_png_pdf(out)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
