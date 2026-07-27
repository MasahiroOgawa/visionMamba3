"""Smoke tests for the CIFAR-10 eval harness (src/eval).

CPU-only, tiny, no dataset needed — exercises model wiring, a 2-step training
loop, and the efficiency probe so the harness stays green in CI.
"""

from __future__ import annotations

import torch

from eval.efficiency import count_params, measure
from eval.models import build_model
from eval.train import WarmupPlateauStrategy, evaluate, set_seed, train_one_epoch


def _fake_loader(n_batches=2, bs=4):
    return [(torch.randn(bs, 3, 32, 32), torch.randint(0, 10, (bs,))) for _ in range(n_batches)]


def test_build_model_shapes_and_params():
    x = torch.randn(2, 3, 32, 32)
    expected = {"vit_attn": 2.6, "vit_mamba3": 2.6, "vit_mamba3_vssd": 2.6}
    for v in expected:
        m = build_model(v, patch_size=4).eval()
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 10)
        assert torch.isfinite(y).all()
        assert count_params(m) / 1e6 > expected[v]  # ~2.7M matched budget


def test_long_sequence_forward():
    """patch_size=1 -> T=1025 tokens; NC-SSD must run (it is the paper's long-T cell)."""
    m = build_model("vit_mamba3_vssd", patch_size=1).eval()
    with torch.no_grad():
        y = m(torch.randn(1, 3, 32, 32))
    assert y.shape == (1, 10) and torch.isfinite(y).all()


def test_two_step_training_reduces_or_runs():
    set_seed(0)
    dev = torch.device("cpu")
    model = build_model("vit_mamba3_vssd", patch_size=4).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    loader = _fake_loader()
    strat = WarmupPlateauStrategy(opt, peak_lr=1e-3, warmup_steps=1, factor=0.5,
                                  patience=1, threshold=1e-3, min_lr=1e-6, ema_alpha=0.3)
    out = train_one_epoch(model, loader, opt, strat, dev, grad_clip=1.0)
    assert torch.isfinite(torch.tensor(out["loss"]))
    ev = evaluate(model, loader, dev)
    assert 0.0 <= ev["acc"] <= 1.0


def test_measure_returns_finite_latency_on_cpu():
    m = build_model("vit_attn", patch_size=4).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        e = measure(lambda inp: m(inp), x, torch.device("cpu"), warmup=1, repeats=2)
    assert e["latency_ms"] > 0.0  # peak_mib is NaN on CPU by design
