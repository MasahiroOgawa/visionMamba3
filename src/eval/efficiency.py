"""Latency / peak-memory measurement (ported from the retired SSM harness's
``bench_efficiency_patched.py``).

``measure`` runs ``warmup`` untimed forwards, then ``repeats`` timed forwards
(median latency) inside a peak-memory scope. Matches how the already-published
Table 1 rows were measured, so NC-SSD numbers are directly comparable.
"""

from __future__ import annotations

import os
import subprocess
import time
from contextlib import contextmanager

import torch


class GpuNotExclusive(RuntimeError):
    """Another process is computing on the GPU, so a timing would be meaningless."""


def assert_gpu_exclusive(device: torch.device) -> None:
    """Refuse to measure while another process holds the GPU.

    A concurrent job does not make a timing noisy, it makes it wrong, and wrong in
    a direction nobody notices: the number still looks plausible. Measured on this
    box, one extra training job stretched a 145.7 s epoch to 322.6 s -- a 2.21x
    slowdown that a latency column would have reported as the operator's cost.

    This is a hard failure rather than a warning because the failure mode it guards
    is silence. Two runs of this grid once overlapped by accident for three minutes;
    had that window contained an efficiency probe, Table 1's latency and memory
    columns would have been wrong with nothing in the output to say so. A warning
    scrolls past in a 12-hour log. Set VM3_ALLOW_SHARED_GPU=1 to override when the
    timing genuinely does not matter (a smoke test, a shape check).
    """
    if device.type != "cuda" or os.environ.get("VM3_ALLOW_SHARED_GPU") == "1":
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return  # No nvidia-smi: nothing to assert against; don't block the run.
    others = [ln.strip() for ln in out.splitlines()
              if ln.strip() and int(ln.split(",")[0]) != os.getpid()]
    if others:
        raise GpuNotExclusive(
            "refusing to measure latency/peak memory: "
            f"{len(others)} other process(es) are computing on this GPU:\n  "
            + "\n  ".join(others)
            + "\nA shared GPU inflates latency (~2.2x per extra job, measured) and "
              "perturbs peak memory. Wait for exclusivity, or set "
              "VM3_ALLOW_SHARED_GPU=1 if this timing is not going to be reported."
        )


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
    assert_gpu_exclusive(device)
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
