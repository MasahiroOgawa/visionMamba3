"""Unit tests for mamba3_attn.mamba3.self_attention.Mamba3SelfAttention"""

from __future__ import annotations

import time

import torch

from visionmamba3.self_attention import Mamba3SelfAttention, ssd_forward
from visionmamba3.mask import build_two_term_mask
from visionmamba3.rope2d import RoPE2D


def test_output_shape_matches_input():
    attn = Mamba3SelfAttention(dim=32, num_heads=4, state_dim=8, bidirectional=False)
    x = torch.randn(2, 16, 32)
    y = attn(x)
    assert y.shape == x.shape


def test_forward_is_differentiable():
    attn = Mamba3SelfAttention(dim=32, num_heads=4, state_dim=8)
    x = torch.randn(1, 8, 32, requires_grad=True)
    y = attn(x)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # And attention params receive gradient too.
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in attn.parameters())
    assert has_grad


def test_bidirectional_equals_forward_plus_gated_reverse():
    # Zero-init `rev_gate` → tanh(0)=0 → reverse stream silenced at init.
    # When we open the gate manually, bidi output should equal
    # forward + tanh(gate) · reverse, per-head.
    torch.manual_seed(42)
    bidi = Mamba3SelfAttention(
        dim=16, num_heads=2, state_dim=4, bidirectional=True, out_proj=False, post_norm=False
    )
    uni = Mamba3SelfAttention(
        dim=16, num_heads=2, state_dim=4, bidirectional=False, out_proj=False, post_norm=False
    )
    # Set gates so tanh maps cleanly to known ratios — different per head to
    # exercise the broadcast.
    with torch.no_grad():
        bidi.rev_gate.copy_(torch.tensor([0.5, 1.2]))
    # Load matching weights (uni has no rev_gate; filter it).
    uni_state = {k: v for k, v in bidi.state_dict().items() if k != "rev_gate"}
    uni.load_state_dict(uni_state, strict=False)

    x = torch.randn(1, 10, 16)
    y_bidi = bidi(x)
    y_fwd = uni(x)
    y_rev = uni(x.flip(dims=(1,))).flip(dims=(1,))

    # y_fwd/y_rev come out post-head-merge as (B, T, H*hd); reshape to (B, T, H, hd)
    # so we can scale per-head, then flatten again.
    B, T, D = y_fwd.shape
    H = 2
    gate = torch.tanh(bidi.rev_gate)  # (H,)
    y_rev_scaled = (y_rev.view(B, T, H, D // H) * gate[None, None, :, None]).reshape(B, T, D)
    assert torch.allclose(y_bidi, y_fwd + y_rev_scaled, atol=1e-5)


def test_bidirectional_zero_init_silences_reverse():
    # At init, rev_gate=0, tanh(0)=0, so bidi == forward-only.
    torch.manual_seed(0)
    bidi = Mamba3SelfAttention(
        dim=16, num_heads=2, state_dim=4, bidirectional=True, out_proj=False, post_norm=False
    )
    uni = Mamba3SelfAttention(
        dim=16, num_heads=2, state_dim=4, bidirectional=False, out_proj=False, post_norm=False
    )
    uni_state = {k: v for k, v in bidi.state_dict().items() if k != "rev_gate"}
    uni.load_state_dict(uni_state, strict=False)
    x = torch.randn(1, 10, 16)
    assert torch.allclose(bidi(x), uni(x), atol=1e-6)


def test_ssd_forward_matches_recurrence_two_term():
    # For two-term mask, verify the matrix form matches the closed-form recurrence:
    # h_t = α_t · h_{t-1} + γ_t · B_t v_t^T    with h_0 = 0
    # y_t = C_t^T · h_t
    torch.manual_seed(0)
    B, H, T, N, D = 1, 1, 6, 3, 4
    Bp = torch.randn(B, H, T, N)
    Cp = torch.randn(B, H, T, N)
    Vp = torch.randn(B, H, T, D)
    delta = torch.rand(B, H, T) * 0.5 + 0.1
    A_log = -(torch.rand(B, H, T) + 0.1)

    L = build_two_term_mask(delta, A_log)
    y_matrix = ssd_forward(Bp, Cp, Vp, L)  # (B, H, T, D)

    alpha = (delta * A_log).exp()
    gamma = delta
    y_rec = torch.zeros_like(y_matrix)
    h = torch.zeros(B, H, N, D)
    for t in range(T):
        h = alpha[:, :, t, None, None] * h + gamma[:, :, t, None, None] * (
            Bp[:, :, t, :, None] * Vp[:, :, t, None, :]
        )
        y_rec[:, :, t, :] = torch.einsum("bhn,bhnd->bhd", Cp[:, :, t, :], h)

    assert torch.allclose(y_matrix, y_rec, atol=1e-4)


def test_linear_time_growth_in_sequence_length():
    # Smoke benchmark: doubling T should not cube the runtime. Generous ceiling.
    attn = Mamba3SelfAttention(dim=64, num_heads=4, state_dim=16, bidirectional=False)
    attn.eval()

    def run(T: int) -> float:
        x = torch.randn(1, T, 64)
        # warmup
        with torch.no_grad():
            attn(x)
        t0 = time.perf_counter()
        for _ in range(3):
            with torch.no_grad():
                attn(x)
        return (time.perf_counter() - t0) / 3

    t64 = run(64)
    t256 = run(256)
    # Quadratic-in-T would be ~16x; SSD should be ≤ ~6x in practice. Loose bound.
    ratio = t256 / max(t64, 1e-6)
    assert ratio < 20, f"sequence growth ratio {ratio:.1f} looks superlinear"


def test_zero_input_gives_zero_output_with_no_bias():
    attn = Mamba3SelfAttention(dim=16, num_heads=2, state_dim=4, bidirectional=True, proj_bias=False)
    # Manually zero out BCNorm biases and projection bias (proj_bias already False).
    for m in attn.modules():
        if hasattr(m, "bias") and isinstance(m.bias, torch.nn.Parameter):
            m.bias.data.zero_()
    x = torch.zeros(2, 5, 16)
    y = attn(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-5)


def test_accepts_rope_without_error():
    rope = RoPE2D()
    attn = Mamba3SelfAttention(dim=16, num_heads=2, state_dim=4, rope=rope)
    x = torch.randn(1, 9, 16)
    pos = torch.cartesian_prod(torch.arange(3), torch.arange(3)).view(1, 9, 2)
    y = attn(x, pos=pos)
    assert y.shape == x.shape
