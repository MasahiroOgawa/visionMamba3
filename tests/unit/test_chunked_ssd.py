"""Chunked SSD must match the full-mask SSD bit-for-bit up to fp32 tolerance.

Covers PLAN §9 R7: at high resolution the T×T mask dominates memory. The
chunked path trades that for O(chunk * T) rows — but the output must be
numerically equivalent, or every Phase B/C checkpoint would silently diverge
from its training-time value at eval time.
"""

from __future__ import annotations

import torch

from visionmamba3.mask import (
    build_three_term_mask,
    build_three_term_mask_rows,
    build_two_term_mask,
    build_two_term_mask_rows,
)
from visionmamba3.self_attention import (
    Mamba3SelfAttention,
    ssd_forward,
    ssd_forward_chunked,
)


def _rand_projections(B: int, H: int, T: int, N: int, D: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    Bp = torch.randn(B, H, T, N, generator=g)
    Cp = torch.randn(B, H, T, N, generator=g)
    Vp = torch.randn(B, H, T, D, generator=g)
    delta = torch.rand(B, H, T, generator=g) * 0.5 + 0.1       # positive
    A_log = -(torch.rand(B, H, T, generator=g) * 0.5 + 0.05)  # negative
    lam = torch.rand(B, H, T, generator=g)                     # [0,1]
    return Bp, Cp, Vp, delta, A_log, lam


def test_two_term_mask_rows_match_full():
    Bsz, H, T = 2, 3, 40
    _, _, _, delta, A_log, _ = _rand_projections(Bsz, H, T, 4, 4)
    full = build_two_term_mask(delta, A_log)
    chunk = 16
    for q0 in range(0, T, chunk):
        q1 = min(q0 + chunk, T)
        rows = build_two_term_mask_rows(delta, A_log, q0, q1)
        assert rows.shape == (Bsz, H, q1 - q0, T)
        torch.testing.assert_close(rows, full[..., q0:q1, :], rtol=1e-6, atol=1e-6)


def test_three_term_mask_rows_match_full():
    Bsz, H, T = 2, 3, 48
    _, _, _, delta, A_log, lam = _rand_projections(Bsz, H, T, 4, 4)
    full = build_three_term_mask(delta, A_log, lam)
    chunk = 16
    for q0 in range(0, T, chunk):
        q1 = min(q0 + chunk, T)
        rows = build_three_term_mask_rows(delta, A_log, lam, q0, q1)
        assert rows.shape == (Bsz, H, q1 - q0, T)
        torch.testing.assert_close(rows, full[..., q0:q1, :], rtol=1e-6, atol=1e-6)


def test_ssd_forward_chunked_matches_full_three_term():
    Bsz, H, T, N, D = 2, 4, 64, 8, 16
    Bp, Cp, Vp, delta, A_log, lam = _rand_projections(Bsz, H, T, N, D)
    L = build_three_term_mask(delta, A_log, lam)
    y_full = ssd_forward(Bp, Cp, Vp, L)
    y_chunked = ssd_forward_chunked(
        Bp, Cp, Vp, delta, A_log, lam,
        three_term=True, chunk_size=16,
    )
    torch.testing.assert_close(y_chunked, y_full, rtol=1e-5, atol=1e-5)


def test_ssd_forward_chunked_matches_full_two_term():
    Bsz, H, T, N, D = 2, 4, 64, 8, 16
    Bp, Cp, Vp, delta, A_log, _ = _rand_projections(Bsz, H, T, N, D)
    L = build_two_term_mask(delta, A_log)
    y_full = ssd_forward(Bp, Cp, Vp, L)
    y_chunked = ssd_forward_chunked(
        Bp, Cp, Vp, delta, A_log, None,
        three_term=False, chunk_size=16,
    )
    torch.testing.assert_close(y_chunked, y_full, rtol=1e-5, atol=1e-5)


def test_ssd_forward_chunked_with_row_renorm():
    Bsz, H, T, N, D = 2, 4, 48, 8, 16
    Bp, Cp, Vp, delta, A_log, lam = _rand_projections(Bsz, H, T, N, D)
    L = build_three_term_mask(delta, A_log, lam)
    y_full = ssd_forward(Bp, Cp, Vp, L, row_renorm=True)
    y_chunked = ssd_forward_chunked(
        Bp, Cp, Vp, delta, A_log, lam,
        three_term=True, row_renorm=True, chunk_size=16,
    )
    torch.testing.assert_close(y_chunked, y_full, rtol=1e-5, atol=1e-5)


def test_module_chunked_matches_full():
    """Full Mamba3SelfAttention with chunk_size set must match chunk_size=None."""
    torch.manual_seed(0)
    dim, heads, T = 64, 4, 48
    attn_full = Mamba3SelfAttention(
        dim=dim, num_heads=heads, state_dim=8,
        bidirectional=True, three_term=True,
        out_proj=False, post_norm=False, chunk_size=None,
        use_fused_kernel=False,
    )
    attn_chunk = Mamba3SelfAttention(
        dim=dim, num_heads=heads, state_dim=8,
        bidirectional=True, three_term=True,
        out_proj=False, post_norm=False, chunk_size=12,
        use_fused_kernel=False,
    )
    attn_chunk.load_state_dict(attn_full.state_dict())
    attn_full.eval(); attn_chunk.eval()

    x = torch.randn(2, T, dim)
    with torch.no_grad():
        y_full = attn_full(x)
        y_chunk = attn_chunk(x)
    torch.testing.assert_close(y_chunk, y_full, rtol=1e-5, atol=1e-5)


def test_chunk_larger_than_T_is_a_noop():
    Bsz, H, T, N, D = 1, 2, 16, 4, 8
    Bp, Cp, Vp, delta, A_log, lam = _rand_projections(Bsz, H, T, N, D)
    L = build_three_term_mask(delta, A_log, lam)
    y_full = ssd_forward(Bp, Cp, Vp, L)
    y_chunk = ssd_forward_chunked(
        Bp, Cp, Vp, delta, A_log, lam,
        three_term=True, chunk_size=1024,
    )
    torch.testing.assert_close(y_chunk, y_full, rtol=1e-6, atol=1e-6)


def test_kernel_path_runs_and_is_close_to_pytorch():
    """Mamba-3 SISO Triton kernel produces structurally similar output to our
    PyTorch SSD path. They are not bit-identical (PLAN § 15.45) — the kernel
    implements the upstream paper's exact formulation while our PyTorch path
    is a from-scratch reimplementation. Cosine similarity ≥ 0.95 on identical
    weights + inputs is the acceptance threshold.
    """
    import pytest
    pytest.importorskip("tilelang", reason="fused kernel requires tilelang")
    if not torch.cuda.is_available():
        pytest.skip("kernel path needs CUDA")
    torch.manual_seed(0)
    common = dict(
        dim=384, num_heads=6, state_dim=64,
        bidirectional=False, three_term=True,
        rope=None, post_norm=False, out_proj=False,
        row_renorm=False, chunk_size=64,
    )
    py = Mamba3SelfAttention(**common, use_fused_kernel=False).cuda().eval()
    kr = Mamba3SelfAttention(**common, use_fused_kernel=True).cuda().eval()
    kr.load_state_dict(py.state_dict())
    x = torch.randn(1, 256, 384, device="cuda")
    with torch.inference_mode():
        y_py = py(x).flatten()
        y_kr = kr(x).flatten()
    cos = torch.nn.functional.cosine_similarity(
        y_py.unsqueeze(0), y_kr.unsqueeze(0)
    ).item()
    assert cos >= 0.95, f"kernel path diverged too far from PyTorch path: cos={cos:.4f}"
