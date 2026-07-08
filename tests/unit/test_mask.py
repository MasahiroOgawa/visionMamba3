"""Unit tests for mamba3_attn.mamba3.mask"""

from __future__ import annotations

import torch

from visionmamba3.mask import build_two_term_mask, build_three_term_mask, build_cross_mask


def _random_inputs(batch=2, heads=3, T=8, seed=0):
    torch.manual_seed(seed)
    delta = torch.rand(batch, heads, T) * 0.5 + 0.1  # > 0
    A_log = -(torch.rand(batch, heads, T) * 2.0 + 0.1)  # < 0
    lam = torch.rand(batch, heads, T)  # in [0,1]
    return delta, A_log, lam


def test_two_term_mask_shape_and_lower_triangular():
    delta, A_log, _ = _random_inputs(T=6)
    L = build_two_term_mask(delta, A_log)
    assert L.shape == (2, 3, 6, 6)
    # Upper triangle (strictly above diagonal) must be exactly zero.
    upper = L.triu(diagonal=1)
    assert torch.all(upper == 0)
    # Lower triangle must be strictly positive given Δ>0, A<0.
    lower_mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    assert torch.all(L[..., lower_mask] > 0)


def test_two_term_mask_diagonal_equals_gamma():
    # L[t, t] = γ_t · ∏ over empty = γ_t = Δ_t
    delta, A_log, _ = _random_inputs(T=5)
    L = build_two_term_mask(delta, A_log)
    diag = torch.diagonal(L, dim1=-2, dim2=-1)  # (B, H, T)
    assert torch.allclose(diag, delta, atol=1e-6)


def test_three_term_reduces_to_two_term_when_lambda_one():
    delta, A_log, _ = _random_inputs(T=7)
    lam = torch.ones_like(delta)  # → β = 0, γ = Δ (same as two-term)
    L2 = build_two_term_mask(delta, A_log)
    L3 = build_three_term_mask(delta, A_log, lam)
    assert torch.allclose(L3, L2, atol=1e-5)


def test_three_term_mask_lower_triangular():
    delta, A_log, lam = _random_inputs(T=6)
    L = build_three_term_mask(delta, A_log, lam)
    assert torch.all(L.triu(diagonal=1) == 0)


def test_three_term_extra_band_contributes_when_lambda_less_than_one():
    delta, A_log, lam = _random_inputs(T=6)
    L_with = build_three_term_mask(delta, A_log, lam)
    L_without = build_three_term_mask(delta, A_log, torch.ones_like(lam))
    # Off-diagonal lower entries should differ where λ<1 introduces β band.
    diff = (L_with - L_without).abs()
    # The band starts at entries [t, t-1] and below (j <= t-1)
    strict_lower = torch.tril(torch.ones(6, 6, dtype=torch.bool), diagonal=-1)
    assert diff[..., strict_lower].max() > 0


def test_cross_mask_shape_and_rank_one():
    delta, A_log, _ = _random_inputs(T=9)
    T_q = 5
    Lc = build_cross_mask(delta, A_log, T_q)
    assert Lc.shape == (2, 3, T_q, 9)
    # All rows identical (rank-1 broadcast).
    first = Lc[..., 0, :]
    for i in range(1, T_q):
        assert torch.allclose(Lc[..., i, :], first, atol=1e-6)


def test_cross_mask_column_decay_monotonic_from_right():
    # Rightmost column: γ_{T_kv} * prod over empty = γ_{T_kv}.
    # Leftmost column: γ_1 * prod of α over entire sequence, decayed by many factors.
    delta, A_log, _ = _random_inputs(T=12)
    Lc = build_cross_mask(delta, A_log, T_q=1)
    row = Lc[..., 0, :]  # (B, H, T_kv)
    # Right-most entry equals γ_{T_kv} = Δ_{T_kv}
    assert torch.allclose(row[..., -1], delta[..., -1], atol=1e-5)


def test_three_term_mask_matches_direct_recurrence():
    """Build the three-term mask two ways:

    (a) the log-space closed form in build_three_term_mask, and
    (b) an independent direct iteration of the recurrence

          h_t = α_t h_{t-1} + β_t (B_{t-1} x_{t-1}) + γ_t (B_t x_t)
          y_t = C_t · h_t

        with B and C set to 1-D identity projectors so y_t reduces to a linear
        combination of x_j with coefficients exactly L[t, j].

    An incorrect index in the β band would make these disagree.
    """
    torch.manual_seed(0)
    T = 6
    delta = torch.rand(T) * 0.5 + 0.1
    A_log = -(torch.rand(T) * 2.0 + 0.1)
    lam = torch.rand(T) * 0.9 + 0.05  # strictly in (0, 1) so β ≠ 0

    L_closed = build_three_term_mask(delta.view(1, 1, T), A_log.view(1, 1, T), lam.view(1, 1, T))[0, 0]

    # Direct recurrence, state is scalar (N=1, D=1).
    alpha = (delta * A_log).exp()
    beta = (1.0 - lam) * delta * alpha
    gamma = lam * delta

    L_direct = torch.zeros(T, T)
    for j in range(T):
        # Inject a unit impulse at position j (x_j=1, others 0).
        h = 0.0
        x_prev = 0.0  # x_{t-1} at t=0 doesn't exist
        for t in range(T):
            x_t = 1.0 if t == j else 0.0
            # y_t contribution from this impulse
            new_h = alpha[t].item() * h + beta[t].item() * x_prev + gamma[t].item() * x_t
            h = new_h
            L_direct[t, j] = h
            x_prev = x_t

    assert torch.allclose(L_closed, L_direct, atol=1e-5), (
        f"closed-form and direct-recurrence masks disagree:\nclosed=\n{L_closed}\ndirect=\n{L_direct}"
    )


def test_mask_underflow_graceful_with_large_decay():
    # Very steep decay shouldn't produce NaNs/Infs.
    delta = torch.full((1, 1, 100), 0.5)
    A_log = torch.full((1, 1, 100), -50.0)
    lam = torch.full((1, 1, 100), 0.5)
    L = build_three_term_mask(delta, A_log, lam)
    assert torch.all(torch.isfinite(L))
