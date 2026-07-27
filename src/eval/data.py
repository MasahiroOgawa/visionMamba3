"""CIFAR-10 data loaders (ported from the retired SSM harness).

Default ``data_dir`` is ``data/cifar10`` — in visionMamba3 that is a symlink to
``~/data/cifar10`` (the dataset was relocated out of the retired SSM repo).
``download=False``: the data is expected to be present already.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_TRAIN_N = 50_000


def make_loaders(batch_size: int, data_dir: Path, num_workers: int, device: torch.device):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_ds = datasets.CIFAR10(root=str(data_dir), train=True, download=False, transform=train_tf)
    test_ds = datasets.CIFAR10(root=str(data_dir), train=False, download=False, transform=test_tf)
    pin = device.type == "cuda"
    persistent = num_workers > 0
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin, drop_last=True, persistent_workers=persistent,
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin, persistent_workers=persistent,
    )
    return train_dl, test_dl
