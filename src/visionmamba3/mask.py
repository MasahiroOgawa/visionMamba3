"""Mamba-3 decay masks.

Given per-token Δ (step), A (negative log-rate), and optionally λ (trapezoidal
weight), construct the lower-triangular decay mask L such that the SSD output
is  Y = (L ⊙ (C Bᵀ)) V.

All masks are built in log-space and exponentiated once at the end to keep
long decays numerically stable.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _log_cumsum_along(x: Tensor, dim: int) -> Tensor:
    return torch.cumsum(x, dim=dim)


def build_two_term_mask(delta: Tensor, A_log: Tensor) -> Tensor:
    """Mamba-2 lower-triangular decay mask.

    α_t = exp(Δ_t · A_t), γ_t = Δ_t
    L[..., t, j] = γ_j · ∏_{k=j+1..t} α_k   for j ≤ t
    L[..., t, j] = 0                         for j > t

    Args:
        delta:  (..., T) positive step sizes
        A_log:  (..., T) negative log-rates (A_t)

    Returns:
        L of shape (..., T, T)
    """
    assert delta.shape == A_log.shape
    T = delta.shape[-1]

    log_alpha = delta * A_log
    log_gamma = torch.log(delta.clamp_min(1e-20))

    # cumulative sum along time (last dim): S_t = Σ_{k=1..t} log α_k
    S = _log_cumsum_along(log_alpha, dim=-1)  # (..., T)

    # Σ_{k=j+1..t} log α_k = S_t - S_j  (S_j includes log α_j)
    log_diff = S.unsqueeze(-1) - S.unsqueeze(-2)  # (..., T, T): entry [t,j] = S_t - S_j

    log_L = log_gamma.unsqueeze(-2) + log_diff  # (..., T, T): entry [t,j] = log γ_j + S_t - S_j

    # Lower-triangular mask; upper triangle -> -inf so exp -> 0
    tri = torch.tril(torch.ones(T, T, device=delta.device, dtype=torch.bool))
    log_L = log_L.masked_fill(~tri, float("-inf"))
    return log_L.exp()


def build_three_term_mask(delta: Tensor, A_log: Tensor, lam: Tensor) -> Tensor:
    """Mamba-3 trapezoidal decay mask.

    With λ_t ∈ [0,1]:
        α_t = exp(Δ_t · A_t)
        β_t = (1 - λ_t) · Δ_t · α_t
        γ_t = λ_t · Δ_t

    L[..., t, j] = (∏_{k=j+1..t} α_k) · γ_j  +  (∏_{k=j+2..t} α_k) · β_{j+1}    for j ≤ t,
                                                                                 (second term is 0 at j=t)
    L[..., t, j] = 0                                                             for j > t

    Args:
        delta: (..., T)  Δ_t > 0
        A_log: (..., T)  A_t < 0
        lam:   (..., T)  λ_t ∈ [0,1]

    Returns:
        L of shape (..., T, T)
    """
    assert delta.shape == A_log.shape == lam.shape
    T = delta.shape[-1]

    log_alpha = delta * A_log
    gamma = lam * delta
    alpha = log_alpha.exp()
    beta = (1.0 - lam) * delta * alpha

    S = _log_cumsum_along(log_alpha, dim=-1)  # S_t = Σ_{k=1..t} log α_k
    log_diff = S.unsqueeze(-1) - S.unsqueeze(-2)  # [t,j] = Σ_{k=j+1..t} log α_k
    tri = torch.tril(torch.ones(T, T, device=delta.device, dtype=torch.bool))

    # First term: (∏_{k=j+1..t} α_k) · γ_j
    first_log = log_diff + torch.log(gamma.clamp_min(1e-20)).unsqueeze(-2)
    first = first_log.masked_fill(~tri, float("-inf")).exp()

    # Second term: (∏_{k=j+2..t} α_k) · β_{j+1}
    # At 0-indexed column k we want beta[k+1] (which is β_{k+2} in 1-indexed =
    # β_{j+1} in the 1-indexed formula). Shift β left by one (drop β_1, append 0).
    beta_shift = torch.nn.functional.pad(beta[..., 1:], (0, 1), value=0.0)  # β_{j+1} at 0-indexed col j
    # Exponent: Σ_{k=j+2..t} log α_k = S_t - S_{j+1}
    S_shift = torch.nn.functional.pad(S[..., 1:], (0, 1), value=0.0)  # S_{j+1} at 0-indexed col j
    second_log = S.unsqueeze(-1) - S_shift.unsqueeze(-2) + torch.log(beta_shift.clamp_min(1e-20)).unsqueeze(-2)
    # Second term is 0 for j == t (since β_{t+1} doesn't exist) and for j > t
    strict_tri = torch.tril(torch.ones(T, T, device=delta.device, dtype=torch.bool), diagonal=-1)
    second = second_log.masked_fill(~strict_tri, float("-inf")).exp()

    return first + second


def build_two_term_mask_rows(
    delta: Tensor, A_log: Tensor, q_start: int, q_end: int
) -> Tensor:
    """Return rows [q_start, q_end) of `build_two_term_mask` without materializing
    the full (T, T) tensor. Peak memory is O((q_end-q_start) * T)."""
    assert delta.shape == A_log.shape
    T = delta.shape[-1]
    device = delta.device

    log_alpha = delta * A_log
    log_gamma = torch.log(delta.clamp_min(1e-20))

    S = _log_cumsum_along(log_alpha, dim=-1)
    S_q = S[..., q_start:q_end]

    log_diff = S_q.unsqueeze(-1) - S.unsqueeze(-2)
    log_L = log_gamma.unsqueeze(-2) + log_diff

    rows = torch.arange(q_start, q_end, device=device)
    cols = torch.arange(T, device=device)
    tri = cols[None, :] <= rows[:, None]
    log_L = log_L.masked_fill(~tri, float("-inf"))
    return log_L.exp()


def build_three_term_mask_rows(
    delta: Tensor, A_log: Tensor, lam: Tensor, q_start: int, q_end: int
) -> Tensor:
    """Return rows [q_start, q_end) of `build_three_term_mask` without materializing
    the full (T, T) tensor. Peak memory is O((q_end-q_start) * T)."""
    assert delta.shape == A_log.shape == lam.shape
    T = delta.shape[-1]
    device = delta.device

    log_alpha = delta * A_log
    gamma = lam * delta
    alpha = log_alpha.exp()
    beta = (1.0 - lam) * delta * alpha

    S = _log_cumsum_along(log_alpha, dim=-1)
    S_q = S[..., q_start:q_end]

    rows = torch.arange(q_start, q_end, device=device)
    cols = torch.arange(T, device=device)
    tri = cols[None, :] <= rows[:, None]
    strict_tri = cols[None, :] < rows[:, None]

    log_diff = S_q.unsqueeze(-1) - S.unsqueeze(-2)
    first_log = log_diff + torch.log(gamma.clamp_min(1e-20)).unsqueeze(-2)
    first = first_log.masked_fill(~tri, float("-inf")).exp()

    beta_shift = torch.nn.functional.pad(beta[..., 1:], (0, 1), value=0.0)
    S_shift = torch.nn.functional.pad(S[..., 1:], (0, 1), value=0.0)
    second_log = (
        S_q.unsqueeze(-1)
        - S_shift.unsqueeze(-2)
        + torch.log(beta_shift.clamp_min(1e-20)).unsqueeze(-2)
    )
    second = second_log.masked_fill(~strict_tri, float("-inf")).exp()

    return first + second


def build_cross_mask(delta_kv: Tensor, A_log_kv: Tensor, T_q: int, bidirectional: bool = False) -> Tensor:
    """Rectangular column-decay mask for cross-attention variant B.

    Unidirectional (default, matches \\eqref{eq:Lcross} in the tex):
        L_cross[..., i, j] = γ_j · ∏_{k=j+1..T_kv} α_k

    Bidirectional (matches \\eqref{eq:Lcross-bi}, added to tex for
    permutation-invariant kv such as images):
        L_cross_bi[..., i, j] = γ_j · [∏_{k=j+1..T_kv} α_k + ∏_{k=1..j-1} α_k]

    The mask is rank-1 across the query axis: all T_q rows are identical
    (the mask is data-dependent on the kv side only).

    Args:
        delta_kv:  (..., T_kv) positive step sizes
        A_log_kv:  (..., T_kv) negative log-rates
        T_q:       number of query tokens
        bidirectional: if True, use the symmetrised bidirectional mask.

    Returns:
        L_cross of shape (..., T_q, T_kv)
    """
    assert delta_kv.shape == A_log_kv.shape
    T_kv = delta_kv.shape[-1]
    log_alpha = delta_kv * A_log_kv
    gamma = delta_kv  # γ_j = Δ_j  (two-term form)

    S = _log_cumsum_along(log_alpha, dim=-1)
    S_total = S[..., -1:]
    fwd = (S_total - S).exp()  # ∏_{k=j+1..T_kv} α_k

    if bidirectional:
        # ∏_{k=1..j-1} α_k = exp(S_{j-1}) with S_0 := 0
        S_prev = torch.nn.functional.pad(S[..., :-1], (1, 0), value=0.0)
        rev = S_prev.exp()
        col_vec = gamma * (fwd + rev)
    else:
        col_vec = gamma * fwd

    return col_vec.unsqueeze(-2).expand(*col_vec.shape[:-1], T_q, T_kv).contiguous()
