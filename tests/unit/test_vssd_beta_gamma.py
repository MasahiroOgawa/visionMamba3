"""Unit tests for visionmamba3.vssd_attention.Mamba3VSSDBetaGammaAttention.

Verifies the operator derived in doc/attention/mamba3_attention.tex
section 6.7.9, eq. (vssd-beta-gamma):

    Y = C1 ((B (*) m1)^T X) + C2 ((B (*) m2)^T X)

The tests that matter here are not the shape ones. The doc's whole argument is
that the second pool buys capacity *only* because its query is independent, and
that a shared query would make it vacuous. Both halves of that claim are
asserted numerically below (test_shared_query_collapses_exactly and
test_independent_query_does_not_collapse), because if the second claim ever
stops holding the operator silently degenerates into VSSD-gamma while still
costing 2x.
"""

from __future__ import annotations

import torch

from visionmamba3.rope2d import RoPE2D
from visionmamba3.vssd_attention import (
    Mamba3VSSDAttention,
    Mamba3VSSDBetaGammaAttention,
    vssd_beta_gamma_forward,
    vssd_forward,
)


def _mk(**kw):
    torch.manual_seed(0)
    return Mamba3VSSDBetaGammaAttention(dim=32, num_heads=4, state_dim=8, **kw)


# --------------------------------------------------------------- basic contract


def test_output_shape_matches_input():
    y = _mk()(torch.randn(2, 16, 32))
    assert y.shape == (2, 16, 32)


def test_forward_is_differentiable_into_both_pools():
    attn = _mk()
    x = torch.randn(1, 8, 32, requires_grad=True)
    attn(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # Pool 2's own parameters must actually receive gradient; if they did not,
    # the extra cost would buy nothing and the tests below would still pass.
    for name in ("proj2.weight", "bc_norm_c2.weight", "A2_bias"):
        p = dict(attn.named_parameters())[name]
        assert p.grad is not None, f"{name} got no gradient"
        assert p.grad.abs().sum() > 0, f"{name} gradient is identically zero"


def test_closed_form_is_the_sum_of_two_gamma_pools():
    torch.manual_seed(0)
    Bsz, H, T, N, Hd = 2, 3, 5, 4, 6
    B = torch.randn(Bsz, H, T, N, dtype=torch.float64)
    C1 = torch.randn(Bsz, H, T, N, dtype=torch.float64)
    C2 = torch.randn(Bsz, H, T, N, dtype=torch.float64)
    V = torch.randn(Bsz, H, T, Hd, dtype=torch.float64)
    m1 = torch.rand(Bsz, H, T, dtype=torch.float64)
    m2 = torch.rand(Bsz, H, T, dtype=torch.float64)

    got = vssd_beta_gamma_forward(B, C1, C2, V, m1, m2)
    want = vssd_forward(B, C1, V, m1) + vssd_forward(B, C2, V, m2)
    assert torch.allclose(got, want)


# ------------------------------------------- the two claims the doc rests on


def test_shared_query_collapses_exactly():
    """With C2 = C1 the two pools factor back into one VSSD-gamma pool.

    C H1 + C H2 = C (H1 + H2) = C ((B (*) (m1+m2))^T X). This is the doc's reason
    the second pool needs its own query, checked numerically rather than trusted.
    """
    torch.manual_seed(0)
    Bsz, H, T, N, Hd = 2, 3, 6, 4, 5
    B = torch.randn(Bsz, H, T, N, dtype=torch.float64)
    C = torch.randn(Bsz, H, T, N, dtype=torch.float64)
    V = torch.randn(Bsz, H, T, Hd, dtype=torch.float64)
    m1 = torch.rand(Bsz, H, T, dtype=torch.float64)
    m2 = torch.rand(Bsz, H, T, dtype=torch.float64)

    two_pools_shared_q = vssd_beta_gamma_forward(B, C, C, V, m1, m2)
    one_pool_combined_m = vssd_forward(B, C, V, m1 + m2)
    assert torch.allclose(two_pools_shared_q, one_pool_combined_m, atol=1e-12)


def test_independent_query_does_not_collapse():
    """With its own C2 the operator leaves VSSD-gamma's expressible set.

    The negation of the test above: no combined weight vector reproduces the
    two-pool output, so the second pool says something the first cannot.
    """
    torch.manual_seed(0)
    Bsz, H, T, N, Hd = 2, 3, 6, 4, 5
    B = torch.randn(Bsz, H, T, N, dtype=torch.float64)
    C1 = torch.randn(Bsz, H, T, N, dtype=torch.float64)
    C2 = torch.randn(Bsz, H, T, N, dtype=torch.float64)
    V = torch.randn(Bsz, H, T, Hd, dtype=torch.float64)
    m1 = torch.rand(Bsz, H, T, dtype=torch.float64)
    m2 = torch.rand(Bsz, H, T, dtype=torch.float64)

    two_pools = vssd_beta_gamma_forward(B, C1, C2, V, m1, m2)
    assert not torch.allclose(two_pools, vssd_forward(B, C1, V, m1 + m2), atol=1e-6)

    # Stronger: least-squares-fit the best single-pool m through C1 and show a
    # residual survives. H = (B (*) m)^T X is linear in m, so this is the best any
    # single pool could do with this query, not merely one unlucky choice.
    best_resid = float("inf")
    for _ in range(200):
        m = torch.rand(Bsz, H, T, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([m], max_iter=40)

        def closure():
            opt.zero_grad()
            loss = (vssd_forward(B, C1, V, m) - two_pools).pow(2).mean()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            best_resid = min(
                best_resid,
                (vssd_forward(B, C1, V, m) - two_pools).pow(2).mean().item(),
            )
        if best_resid < 1e-10:
            break
    scale = two_pools.pow(2).mean().item()
    assert best_resid / scale > 1e-3, (
        f"a single pool fitted the two-pool output to relative residual "
        f"{best_resid / scale:.2e}; the second pool is adding nothing"
    )


# -------------------------------------------------------- non-causality, masking


def test_output_is_non_causal():
    """Token 0's output must respond to the last token: no causal masking."""
    attn = _mk().double()
    x = torch.randn(1, 10, 32, dtype=torch.float64)
    y0 = attn(x)
    x2 = x.clone()
    x2[0, -1] += 5.0
    y1 = attn(x2)
    assert not torch.allclose(y0[0, 0], y1[0, 0], atol=1e-8)


def test_attn_mask_excludes_tokens_from_both_pools():
    """A masked token must not reach any query, through either pool."""
    attn = _mk().double()
    x = torch.randn(1, 8, 32, dtype=torch.float64)
    keep = torch.ones(1, 8, dtype=torch.bool)
    keep[0, 5] = False

    y_masked = attn(x, attn_mask=keep)
    x2 = x.clone()
    x2[0, 5] += 100.0            # perturb only the masked token
    y_masked2 = attn(x2, attn_mask=keep)
    # Masked token's own output may move (its query still reads the pools);
    # every other token's must not.
    other = [i for i in range(8) if i != 5]
    assert torch.allclose(y_masked[0, other], y_masked2[0, other], atol=1e-9)


# --------------------------------------------------- positional-encoding wiring


def test_pool2_query_is_rotated_like_b():
    """The whole operator's output must depend only on *relative* position.

    Shifting every token's grid position by a constant leaves all pairwise offsets
    unchanged, so a correctly-wired operator returns the identical output. This
    holds only if B, C1 *and* C2 are rotated alike: leaving C2 unrotated, or
    rotating it in different channel pairs, leaks absolute position into pool 2's
    scores and the shift moves the output.

    Regression guard for the failure that already cost the 2-directional operator
    12 points once, in the one place a second query stream could reintroduce it.
    """
    attn = _mk(rope=RoPE2D()).double()
    x = torch.randn(1, 9, 32, dtype=torch.float64)
    grid = torch.arange(9)
    pos = torch.stack([grid // 3, grid % 3], dim=-1)[None]

    y_here = attn(x, pos=pos)
    y_shifted = attn(x, pos=pos + 4)
    assert torch.allclose(y_here, y_shifted, atol=1e-8), (
        "output changed under a global position shift: some query is not rotated "
        "the same way as B"
    )


def test_rope_and_rotary_run_and_change_output():
    """Both positional encodings are actually wired into this variant."""
    x = torch.randn(2, 9, 32)
    # RoPE2D wants (B, T, 2) integer (y, x) coords -- a 3x3 grid of 9 tokens.
    grid = torch.arange(9)
    pos = torch.stack([grid // 3, grid % 3], dim=-1)[None].expand(2, -1, -1)

    plain = _mk()(x)
    with_rope = _mk(rope=RoPE2D())(x, pos=pos)
    with_rotary = _mk(rope_angles=True)(x)

    assert with_rope.shape == plain.shape
    assert not torch.allclose(plain, with_rope, atol=1e-6)
    # The rotary head is zero-initialised, so at init it is the identity; the
    # projection width still changes, which is what we can assert cheaply here.
    assert with_rotary.shape == plain.shape
    assert _mk(rope_angles=True).projections.num_rope_angles == 4


# ------------------------------------------------------------------ cost, swap-in


def test_costs_one_extra_query_and_scalar_against_gamma():
    gamma = Mamba3VSSDAttention(dim=32, num_heads=4, state_dim=8)
    bg = Mamba3VSSDBetaGammaAttention(dim=32, num_heads=4, state_dim=8)
    n_gamma = sum(p.numel() for p in gamma.parameters())
    n_bg = sum(p.numel() for p in bg.parameters())
    H, N = 4, 8
    expected_extra = (
        H * N * 32 + H * 32   # proj2: C2 rows + A2 row
        + 2 * H * N           # bc_norm_c2 weight + bias
        + H                   # A2_bias
    )
    assert n_bg - n_gamma == expected_extra


def test_accepts_the_same_kwargs_as_the_other_operators():
    """install_mamba3 passes one kwarg set regardless of variant."""
    attn = Mamba3VSSDBetaGammaAttention(
        dim=32, num_heads=4, state_dim=8, bidirectional=True, three_term=True,
        row_renorm=True, chunk_size=64, use_fused_kernel=True,
        out_proj=True, proj_bias=False, post_norm=True,
    )
    assert attn(torch.randn(1, 5, 32)).shape == (1, 5, 32)


def test_cpu_and_cuda_agree():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    attn = Mamba3VSSDBetaGammaAttention(dim=32, num_heads=4, state_dim=8).double()
    x = torch.randn(2, 7, 32, dtype=torch.float64)
    y_cpu = attn(x)
    y_cuda = attn.cuda()(x.cuda()).cpu()
    assert torch.allclose(y_cpu, y_cuda, atol=1e-9)
