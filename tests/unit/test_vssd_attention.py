"""Unit tests for mamba3_attn.mamba3.vssd_attention.Mamba3VSSDAttention.

Verifies the NC-SSD operator derived in doc/attention/mamba3_attention.tex §6:
    Y = C · ((B ⊙ m)ᵀ X)
"""

from __future__ import annotations

import torch

from visionmamba3.vssd_attention import Mamba3VSSDAttention, vssd_forward
from visionmamba3.rope2d import RoPE2D


def test_output_shape_matches_input():
    attn = Mamba3VSSDAttention(dim=32, num_heads=4, state_dim=8)
    x = torch.randn(2, 16, 32)
    y = attn(x)
    assert y.shape == x.shape


def test_forward_is_differentiable():
    attn = Mamba3VSSDAttention(dim=32, num_heads=4, state_dim=8)
    x = torch.randn(1, 8, 32, requires_grad=True)
    y = attn(x)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in attn.parameters())
    assert has_grad


def test_vssd_forward_matches_closed_form():
    """vssd_forward implements Y = C((B ⊙ m)ᵀ X) per head."""
    torch.manual_seed(0)
    Bsz, H, T, N, Hd = 2, 3, 5, 4, 6
    B = torch.randn(Bsz, H, T, N)
    C = torch.randn(Bsz, H, T, N)
    V = torch.randn(Bsz, H, T, Hd)
    m = torch.rand(Bsz, H, T)

    y = vssd_forward(B, C, V, m)

    # Reference: build the global hidden state H = Σ_t m_t · B_t ⊗ V_t,
    # then y_i = C_iᵀ H.
    ref = torch.zeros(Bsz, H, T, Hd)
    for b in range(Bsz):
        for h in range(H):
            state = torch.zeros(N, Hd)
            for t in range(T):
                state += m[b, h, t] * torch.outer(B[b, h, t], V[b, h, t])
            for i in range(T):
                ref[b, h, i] = C[b, h, i] @ state
    assert torch.allclose(y, ref, atol=1e-5)


def test_nc_ssd_is_non_causal():
    """Token 0's output must depend on token T-1's input (gradient through
    the full T-axis), unlike a causal kernel."""
    torch.manual_seed(0)
    attn = Mamba3VSSDAttention(dim=16, num_heads=2, state_dim=4, post_norm=False, out_proj=False)
    T = 6
    x = torch.randn(1, T, 16, requires_grad=True)
    y = attn(x)
    # ∂y[0] / ∂x[T-1] should be non-zero.
    grad = torch.autograd.grad(y[0, 0].sum(), x, retain_graph=False)[0]
    assert grad[0, -1].abs().sum() > 0, "NC-SSD must propagate gradient from last token to first"


def test_2d_rope_changes_output_for_same_token():
    """Two identical tokens at different (y,x) positions must produce different
    outputs when RoPE is wired in — the only signal distinguishing them is the
    rotary embedding applied to B and C."""
    torch.manual_seed(0)
    attn = Mamba3VSSDAttention(
        dim=32, num_heads=4, state_dim=8, rope=RoPE2D(base_frequency=10.0),
        post_norm=False, out_proj=False,
    )
    tok = torch.randn(1, 1, 32)
    x = tok.expand(1, 3, 32).contiguous()      # 3 identical tokens
    pos_a = torch.tensor([[[0, 0], [0, 1], [0, 2]]])  # row 0, three columns
    pos_b = torch.tensor([[[0, 0], [1, 0], [2, 0]]])  # column 0, three rows
    ya = attn(x, pos=pos_a)
    yb = attn(x, pos=pos_b)
    assert not torch.allclose(ya, yb, atol=1e-5), "RoPE must distinguish row vs column positions"


def test_attn_mask_zeros_out_tokens():
    """attn_mask=(B,T) with a 0 at position j must remove j's contribution to
    the global hidden state, so all outputs change vs. the unmasked run."""
    torch.manual_seed(0)
    attn = Mamba3VSSDAttention(dim=16, num_heads=2, state_dim=4, post_norm=False, out_proj=False)
    x = torch.randn(1, 5, 16)
    keep_all = torch.ones(1, 5)
    keep_drop = torch.tensor([[1, 1, 0, 1, 1]], dtype=torch.float)
    y_all = attn(x, attn_mask=keep_all)
    y_drop = attn(x, attn_mask=keep_drop)
    assert not torch.allclose(y_all, y_drop)


def test_cpu_path_runs():
    """NC-SSD has no Triton dependency — pure PyTorch must work on CPU."""
    attn = Mamba3VSSDAttention(dim=16, num_heads=2, state_dim=4)
    x = torch.randn(2, 4, 16)
    y = attn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
