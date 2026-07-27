"""Training / eval loop + LR strategies (ported from the retired SSM harness).

Recipe is kept identical: AdamW (wd 0.05), per-step linear warmup, then either
cosine decay or ReduceLROnPlateau on an EMA of train loss; bf16 autocast on CUDA;
top-1 accuracy via argmax.
"""

from __future__ import annotations

import contextlib
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def autocast_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _make_scheduler(optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


class WarmupCosineStrategy:
    """Per-step linear warmup -> cosine decay."""

    def __init__(self, optimizer, total_steps: int, warmup_steps: int):
        self.optimizer = optimizer
        self.scheduler = _make_scheduler(optimizer, total_steps, warmup_steps)

    def step_per_step(self) -> None:
        self.scheduler.step()

    def step_per_epoch(self, train_loss: float) -> None:
        pass

    @property
    def last_lr(self) -> float:
        return self.scheduler.get_last_lr()[0]


class WarmupPlateauStrategy:
    """Per-step linear warmup, then per-epoch ReduceLROnPlateau on EMA(train_loss)."""

    def __init__(self, optimizer, peak_lr: float, warmup_steps: int,
                 factor: float, patience: int, threshold: float, min_lr: float,
                 ema_alpha: float):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.ema_alpha = ema_alpha
        self.step_count = 0
        self.loss_ema: float | None = None
        self.plateau = ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience,
            threshold=threshold, min_lr=min_lr,
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = 0.0

    def step_per_step(self) -> None:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.peak_lr * self.step_count / max(1, self.warmup_steps)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def step_per_epoch(self, train_loss: float) -> None:
        if self.loss_ema is None:
            self.loss_ema = train_loss
        else:
            self.loss_ema = self.ema_alpha * train_loss + (1.0 - self.ema_alpha) * self.loss_ema
        if self.step_count > self.warmup_steps:
            self.plateau.step(self.loss_ema)

    @property
    def last_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def train_one_epoch(model, loader, optimizer, lr_strategy, device, grad_clip: float = 0.0):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    t0 = time.perf_counter()
    last_lr = lr_strategy.last_lr
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        lr_strategy.step_per_step()
        last_lr = lr_strategy.last_lr
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(-1) == y).sum().item()
        total_n += bs
    return {
        "loss": total_loss / max(1, total_n),
        "acc": total_correct / max(1, total_n),
        "lr_end": last_lr,
        "wall_s": time.perf_counter() - t0,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_ctx(device):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(-1) == y).sum().item()
        total_n += bs
    return {"loss": total_loss / max(1, total_n), "acc": total_correct / max(1, total_n)}
