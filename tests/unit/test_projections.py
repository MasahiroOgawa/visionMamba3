"""Unit tests for mamba3_attn.mamba3.projections"""

from __future__ import annotations

import torch

from visionmamba3.projections import AttentionProjections, BCNorm


def test_projections_shapes():
    proj = AttentionProjections(dim=32, num_heads=4, state_dim=8)
    x = torch.randn(2, 10, 32)
    B_t, C_t, V_t, delta, A_log, lam = proj(x)
    assert B_t.shape == (2, 4, 10, 8)
    assert C_t.shape == (2, 4, 10, 8)
    assert V_t.shape == (2, 4, 10, 8)  # head_dim = 32/4 = 8
    assert delta.shape == (2, 4, 10)
    assert A_log.shape == (2, 4, 10)
    assert lam.shape == (2, 4, 10)


def test_projection_sign_constraints():
    proj = AttentionProjections(dim=16, num_heads=2, state_dim=4)
    x = torch.randn(3, 20, 16) * 3  # larger scale to stress-test
    _, _, _, delta, A_log, lam = proj(x)
    assert torch.all(delta > 0)
    assert torch.all(A_log < 0)
    assert torch.all(lam >= 0) and torch.all(lam <= 1)


def test_projection_is_differentiable():
    proj = AttentionProjections(dim=16, num_heads=2, state_dim=4)
    x = torch.randn(1, 5, 16, requires_grad=True)
    B_t, C_t, V_t, delta, A_log, lam = proj(x)
    loss = B_t.sum() + C_t.sum() + V_t.sum() + delta.sum() + A_log.sum() + lam.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_bcnorm_unit_rms():
    norm = BCNorm(num_heads=4, state_dim=16)
    x = torch.randn(2, 4, 10, 16) * 5
    y = norm(x)
    # After norm with weight=1, bias=0 (init), RMS over last dim should be ~1.
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_bcnorm_has_learnable_weight_and_bias():
    norm = BCNorm(num_heads=3, state_dim=5)
    assert norm.weight.requires_grad and norm.weight.shape == (3, 5)
    assert norm.bias.requires_grad and norm.bias.shape == (3, 5)
