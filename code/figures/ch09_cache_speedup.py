"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 9 — Cache/precompute speedup demo (SVG).

Output: figures/ch09_cache_speedup.svg
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import torch
from torch.utils.data import Dataset, DataLoader

class SlowOnTheFly(Dataset):
    def __init__(self, n=600, delay=0.0004):
        self.n=n; self.delay=delay
    def __len__(self): return self.n
    def __getitem__(self, i):
        time.sleep(self.delay)
        x = torch.tensor([(i%10)/10.0, ((i*7)%10)/10.0], dtype=torch.float32)
        y = torch.tensor((i%2), dtype=torch.long)
        return x, y

class Precomputed(Dataset):
    def __init__(self, n=600, delay=0.0004):
        xs=[]; ys=[]
        for i in range(n):
            time.sleep(delay)
            xs.append(torch.tensor([(i%10)/10.0, ((i*7)%10)/10.0], dtype=torch.float32))
            ys.append(torch.tensor((i%2), dtype=torch.long))
        self.X=torch.stack(xs); self.Y=torch.stack(ys)
    def __len__(self): return self.X.size(0)
    def __getitem__(self, i): return self.X[i], self.Y[i]

def epoch_time(ds: Dataset) -> float:
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    t0=time.time()
    for _ in loader: pass
    return time.time()-t0

def main() -> None:
    out = Path('figures/ch09_cache_speedup.svg'); out.parent.mkdir(parents=True, exist_ok=True)
    # Both datasets use the same synthetic per-item delay; precomputed pays it once
    # up-front during construction, so per-epoch time is lower but not extreme.
    t1 = epoch_time(SlowOnTheFly(n=600, delay=0.0004))
    t2 = epoch_time(Precomputed(n=600, delay=0.0004))
    plt.figure(figsize=(5.6,3.0))
    plt.bar(['on-the-fly','precomputed'], [t1, t2])
    plt.ylabel('epoch time (s)'); plt.title('Caching heavy transforms speeds up I/O')
    plt.tight_layout(); plt.savefig(out, format='svg')
    print(f"Wrote {out}")

if __name__=='__main__':
    main()
