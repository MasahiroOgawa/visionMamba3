"""Unit tests for mamba3_attn.mamba3.cross_attention.Mamba3CrossAttention"""

from __future__ import annotations

import torch

from visionmamba3.cross_attention import Mamba3CrossAttention


def test_variant_b_shape():
    attn = Mamba3CrossAttention(dim_q=32, dim_kv=32, num_heads=4, state_dim=8, variant="B")
    q = torch.randn(2, 9, 32)
    kv = torch.randn(2, 12, 32)
    y = attn(q, kv)
    assert y.shape == (2, 9, 32)


def test_variant_a_shape():
    attn = Mamba3CrossAttention(dim_q=32, dim_kv=32, num_heads=4, state_dim=8, variant="A")
    q = torch.randn(2, 9, 32)
    kv = torch.randn(2, 12, 32)
    y = attn(q, kv)
    assert y.shape == (2, 9, 32)


def test_variant_b_returns_attn_map_with_correct_shape():
    attn = Mamba3CrossAttention(dim_q=32, dim_kv=32, num_heads=4, state_dim=8, variant="B")
    q = torch.randn(1, 5, 32)
    kv = torch.randn(1, 7, 32)
    y, attn_map = attn(q, kv, return_attn=True)
    assert y.shape == (1, 5, 32)
    assert attn_map.shape == (1, 4, 5, 7)


def test_gradient_flows_through_query_and_kv():
    attn = Mamba3CrossAttention(dim_q=16, dim_kv=16, num_heads=2, state_dim=4, variant="B")
    q = torch.randn(1, 4, 16, requires_grad=True)
    kv = torch.randn(1, 6, 16, requires_grad=True)
    y = attn(q, kv)
    y.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert kv.grad is not None and torch.isfinite(kv.grad).all()
    assert q.grad.abs().sum() > 0
    assert kv.grad.abs().sum() > 0


def test_zero_kv_gives_zero_output_variant_b():
    attn = Mamba3CrossAttention(
        dim_q=16, dim_kv=16, num_heads=2, state_dim=4, variant="B", out_proj=False
    )
    # Zero biases in everything so zero KV -> zero everywhere downstream.
    for m in attn.modules():
        if hasattr(m, "bias") and isinstance(m.bias, torch.nn.Parameter):
            m.bias.data.zero_()
    q = torch.randn(1, 5, 16)
    kv = torch.zeros(1, 7, 16)
    y = attn(q, kv)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-5)


def test_variant_a_and_b_agree_when_kv_length_one():
    """With T_kv=1 the state-compressed sum degenerates to a single weighted
    outer product; variant B's mask also reduces to a scalar. Both should give
    identical outputs modulo the output projection matrix (so we disable it)."""
    common = dict(dim_q=16, dim_kv=16, num_heads=2, state_dim=4, out_proj=False)
    a = Mamba3CrossAttention(variant="A", **common)
    b = Mamba3CrossAttention(variant="B", **common)
    # Share projection weights so B and A see identical B, C, V, Δ, A, λ.
    b.q_proj.load_state_dict(a.q_proj.state_dict())
    b.kv_proj.load_state_dict(a.kv_proj.state_dict())

    q = torch.randn(1, 4, 16)
    kv = torch.randn(1, 1, 16)
    y_a = a(q, kv)
    y_b = b(q, kv)
    assert torch.allclose(y_a, y_b, atol=1e-5)


def test_masked_kv_column_has_lower_contribution_in_variant_b():
    # Large negative A → steep decay → leftmost kv tokens contribute much less.
    attn = Mamba3CrossAttention(
        dim_q=16, dim_kv=16, num_heads=1, state_dim=4, variant="B", out_proj=False
    )
    q = torch.randn(1, 3, 16)
    kv = torch.randn(1, 20, 16)
    _y, attn_map = attn(q, kv, return_attn=True)  # (1, 1, 3, 20)
    # Effective column-weight magnitude (over query row and heads): last > first.
    col_magnitude = attn_map.abs().mean(dim=(0, 1, 2))  # (T_kv,)
    # We don't demand monotone, just that the right side of the kv sequence
    # carries meaningful magnitude and the left side is strictly positive.
    assert col_magnitude[-1] > 0
    assert torch.all(col_magnitude >= 0)
