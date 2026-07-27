"""Latency / peak-memory measurement (ported from the retired SSM harness's
``bench_efficiency_patched.py``).

``measure`` runs ``warmup`` untimed forwards, then ``repeats`` timed forwards
(median latency) inside a peak-memory scope. Matches how the already-published
Table 1 rows were measured, so NC-SSD numbers are directly comparable.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch


@contextmanager
def cuda_mem_scope(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(forward_fn, x: torch.Tensor, device: torch.device,
            warmup: int = 2, repeats: int = 5) -> dict:
    for _ in range(warmup):
        forward_fn(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    with cuda_mem_scope(device):
        timings_ms: list[float] = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            forward_fn(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
    timings_ms.sort()
    median = timings_ms[len(timings_ms) // 2]
    peak = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda" else float("nan")
    )
    return {"latency_ms": median, "peak_mib": peak}


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
