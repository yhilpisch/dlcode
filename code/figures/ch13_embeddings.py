"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.

Chapter 13 — Toy embedding scatter (SVG).

Output: figures/ch13_embeddings.svg
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

def main() -> None:
    out = Path('figures/ch13_embeddings.svg'); out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    words = ['good','great','bad','awful','movie','film']
    base = np.array([[ 1.0,  1.0], [ 1.1,  0.9], [-1.0, -1.0], [-1.1, -0.9], [0.1, 0.0], [0.0, 0.1]])
    pts = base + 0.08*rng.randn(len(words),2)
    fig, ax = plt.subplots(figsize=(5.6,4.0))
    ax.scatter(pts[:,0], pts[:,1], c=['tab:green','tab:green','tab:red','tab:red','tab:blue','tab:blue'])
    for (x,y),w in zip(pts,words): ax.text(x+0.02,y+0.02,w, fontsize=10)
    ax.set_title('Toy embedding space (2D)'); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, format='svg'); print(f"Wrote {out}")

if __name__ == '__main__':
    main()

