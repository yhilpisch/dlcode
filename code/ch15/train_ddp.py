"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Minimal DDP training template with AMP, gradient accumulation, and resumable
checkpoints. Launch with torchrun, e.g.:

  torchrun --standalone --nproc_per_node=2 code/ch15/train_ddp.py \
    --epochs 2 --batch 128 --accum 2 --amp

The model/dataset are tiny so you can verify the wiring without a large GPU.
Replace `TinyNet` and the synthetic dataset with your actual code.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, distributed


@dataclass
class Config:
    epochs: int = 2
    batch: int = 128
    lr: float = 3e-4
    accum: int = 1
    amp: bool = False
    ckpt: Path = Path("checkpoints/ch15_ddp.pt")


class ToyDataset(Dataset):
    def __init__(self, n: int = 10_000, d: int = 64, *, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, d, generator=g)
        w = torch.randn(d, generator=g)
        y = (self.X @ w + 0.1 * torch.randn(n, generator=g)).gt(0).long()
        self.y = y

    def __len__(self) -> int:  # type: ignore[override]
        return self.X.size(0)

    def __getitem__(
        self, i: int
    ) -> Tuple[torch.Tensor, int]:  # type: ignore[override]
        return self.X[i], int(self.y[i])


class TinyNet(nn.Module):
    def __init__(self, d: int = 64, h: int = 128, k: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Linear(h, k),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def _distributed_env_present() -> bool:
    return all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))


def setup_ddp() -> None:
    if dist.is_initialized() or not _distributed_env_present():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def is_main() -> bool:
    return rank() == 0


def save_ckpt(
    path: Path,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    epoch: int,
) -> None:
    to_save = model.module if isinstance(model, DDP) else model
    if is_main():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": to_save.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch,
            },
            path,
        )
    if dist.is_initialized():
        dist.barrier()


def load_ckpt(path: Path, model: nn.Module, opt: torch.optim.Optimizer) -> int:
    if not path.exists():
        return 0
    state = torch.load(path, map_location="cpu")
    to_load = model.module if isinstance(model, DDP) else model
    to_load.load_state_dict(state["model"])  # type: ignore[arg-type]
    opt.load_state_dict(state["opt"])  # type: ignore[arg-type]
    return int(state.get("epoch", 0)) + 1


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        device = next(model.parameters()).device
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / max(total, 1)


def train(cfg: Config) -> None:
    setup_ddp()

    ds_train = ToyDataset(n=20_000)
    ds_val = ToyDataset(n=4_000, seed=1)

    sampler = (
        distributed.DistributedSampler(ds_train) if dist.is_initialized() else None
    )
    loader = DataLoader(
        ds_train,
        batch_size=cfg.batch,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=True,
    )
    loader_val = DataLoader(ds_val, batch_size=1024, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyNet().to(device)
    if dist.is_initialized():
        if device.type == "cuda":
            model = DDP(model, device_ids=[torch.cuda.current_device()])
        else:
            model = DDP(model)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(
        enabled=(cfg.amp and device.type == "cuda")
    )

    start_epoch = load_ckpt(cfg.ckpt, model, opt)

    for epoch in range(start_epoch, cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)
        for it, (xb, yb) in enumerate(loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(
                enabled=(cfg.amp and device.type == "cuda")
            ):
                loss = loss_fn(model(xb), yb) / max(cfg.accum, 1)
            scaler.scale(loss).backward()
            if (it + 1) % cfg.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            running += float(loss) * max(cfg.accum, 1)

        # eval on rank 0 for brevity
        if is_main():
            acc = evaluate(
                model.module if isinstance(model, DDP) else model, loader_val
            )
            print(
                f"epoch {epoch}: loss={running/(it + 1):.4f} val_acc={acc:.3f}"
            )
        save_ckpt(cfg.ckpt, model, opt, epoch)

    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--amp", action="store_true")
    p.add_argument(
        "--ckpt", type=Path, default=Path("checkpoints/ch15_ddp.pt")
    )
    a = p.parse_args()
    return Config(
        epochs=a.epochs,
        batch=a.batch,
        lr=a.lr,
        accum=a.accum,
        amp=a.amp,
        ckpt=a.ckpt,
    )


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    cfg = parse_args()
    train(cfg)
