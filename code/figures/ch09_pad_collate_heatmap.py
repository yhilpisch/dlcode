"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 9 — Pad collate heatmap (SVG).

Output: figures/ch09_pad_collate_heatmap.svg
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import torch
from torch.utils.data import Dataset, DataLoader
from code.figures._save import save_png_pdf

class ToySeq(Dataset):
    def __init__(self, rng, n: int = 12) -> None:
        self.x = [
            torch.tensor(
                rng.integers(1, 10, size=rng.integers(3, 8))
            )
            for _ in range(n)
        ]
        self.y = [int(x.sum() % 2) for x in self.x]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i].float(), self.y[i]

def pad_collate(batch):
    xs, ys = zip(*batch)
    L = max(x.size(0) for x in xs)
    Xp = torch.zeros(len(xs), L)
    for i, x in enumerate(xs):
        Xp[i, : x.size(0)] = x
    return Xp, torch.tensor(ys, dtype=torch.long)

def main() -> None:
    out = Path("figures/ch09_pad_collate_heatmap.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    loader = DataLoader(
        ToySeq(rng, n=16), batch_size=6, collate_fn=pad_collate
    )
    xb, yb = next(iter(loader))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    im = ax.imshow(xb.numpy(), cmap="viridis")
    ax.set_xlabel("time step")
    ax.set_ylabel("batch item")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    save_png_pdf(out)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
