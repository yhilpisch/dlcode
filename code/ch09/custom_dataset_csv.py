"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 9 — Example custom Dataset that reads a CSV.

Note: This is a listing for Appendix F; it demonstrates structure without
requiring a CSV in the repo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    """A tiny CSV dataset: last column is an integer label; others are features.

    Expects a headerless CSV with numeric values.
    """

    def __init__(self, csv_path: str | Path, transform=None) -> None:
        self.path = Path(csv_path)
        arr = np.loadtxt(self.path, delimiter=",")
        self.X = torch.tensor(arr[:, :-1], dtype=torch.float32)
        self.y = torch.tensor(arr[:, -1], dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        x = self.X[i]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[i]


if __name__ == "__main__":
    # Example usage (requires a CSV path):
    # ds = CSVDataset('data/my_tiny.csv')
    # print(len(ds), ds[0])
    pass
