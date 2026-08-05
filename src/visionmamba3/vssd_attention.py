"""Mamba-3-flavored VSSD (Non-Causal SSD) self-attention.

Implements the operator derived in `doc/attention/mamba3_attention.tex §6`:

    Y = C · ((B ⊙ m)ᵀ X)              [eq. (nc-Y)]

where m_t > 0 is a per-token scalar that replaces the T×T structured mask L
of full Mamba-3 SSD. Compared with :class:`Mamba3SelfAttention`:

* No T×T mask — peak memory O(N_state · head_dim) per head, independent of T.
* No forward+reverse scan — a single forward einsum yields non-causal output.
* No fused Triton kernel needed — two cheap einsums run anywhere PyTorch does
  (CUDA and CPU paths are identical).

We reuse :class:`AttentionProjections` so the parameter inventory and weight
shapes match :class:`Mamba3SelfAttention` exactly. From the 6-output bundle
``(B, C, V, Δ, A_log, λ)`` only ``B, C, V, A_log`` are used; ``Δ`` and ``λ``
do not survive the NC reduction (see doc §6.4). ``m`` is derived as
``softplus(A_log_raw)``: VSSD §3.2 notes that ``A`` and ``1/A`` share the
same range, so we learn ``m`` directly through the existing scalar projection.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .projections import AttentionProjections, BCNorm
from .self_attention import apply_cumulative_rope


def vssd_forward(B: Tensor, C: Tensor, V: Tensor, m: Tensor) -> Tensor:
    """Apply the NC-SSD output formula.

    Args:
        B: (batch, H, T, N_state) — key projection (post-BCNorm, post-RoPE).
        C: (batch, H, T, N_state) — query projection (post-BCNorm, post-RoPE).
        V: (batch, H, T, head_dim) — value (post-SiLU).
        m: (batch, H, T) — strictly positive per-token weight.

    Returns:
        Y: (batch, H, T, head_dim).
    """
    Bm = B * m.unsqueeze(-1)                          # (B, H, T, N)
    H = torch.einsum("bhtn,bhtd->bhnd", Bm, V)        # (B, H, N, head_dim)
    return torch.einsum("bhtn,bhnd->bhtd", C, H)      # (B, H, T, head_dim)


class Mamba3VSSDAttention(nn.Module):
    """Mamba-3 NC-SSD (VSSD) self-attention.

    Constructor mirrors :class:`Mamba3SelfAttention` for swap-in compatibility.
    Flags that don't apply to NC-SSD (``bidirectional``, ``three_term``,
    ``chunk_size``, ``use_fused_kernel``) are accepted-and-ignored so callers
    can switch operators without changing keyword arguments.

    Args:
        dim: token feature dimension D.
        num_heads: H (D must be divisible by H).
        state_dim: N_state per head (default 64).
        rope: optional module ``forward(tokens, positions) -> tokens`` applied
            to B and C before the einsum (doc §6.6 item 3).
        out_proj, proj_bias, post_norm: same as :class:`Mamba3SelfAttention`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        state_dim: int = 64,
        rope: Optional[nn.Module] = None,
        out_proj: bool = True,
        proj_bias: bool = True,
        post_norm: bool = True,
        # Accepted-and-ignored — kept to mirror Mamba3SelfAttention's signature
        # so install_mamba3 can pass the same kwargs regardless of variant.
        bidirectional: bool = True,
        three_term: bool = True,
        rope_angles: bool = False,
        row_renorm: bool = True,
        chunk_size: Optional[int] = None,
        use_fused_kernel: bool = True,
    ) -> None:
        super().__init__()
        del bidirectional, three_term, row_renorm, chunk_size, use_fused_kernel

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.state_dim = state_dim
        self.rope = rope

        self.projections = AttentionProjections(
            dim, num_heads, state_dim, rope_angles=rope_angles
        )
        self.post_norm = nn.LayerNorm(dim) if post_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias) if out_proj else nn.Identity()

    def forward(
        self,
        x: Tensor,
        pos: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: (B, T, D).
            pos: (B, T, 2) integer 2-D positions for RoPE, or None.
            attn_mask: (B, T) boolean/0-1 — True/1 keeps the token. When given,
                masked tokens are zeroed before they enter the global hidden
                state H, so every query reads only from unmasked tokens.

        Returns:
            y: (B, T, D).
        """
        B_t, C_t, V_t, m, _rotate = self._project(x, pos, attn_mask)
        return self._finish(vssd_forward(B_t, C_t, V_t, m))

    # -- shared front/back halves, reused by Mamba3VSSDBetaGammaAttention ------

    def _project(self, x: Tensor, pos: Optional[Tensor], attn_mask: Optional[Tensor]):
        """Project, apply both positional encodings, and build m.

        Returns ``(B, C, V, m, rotate)``. ``rotate`` replays the *same* rotations
        B and C received, for operators that add a second query stream. Handing it
        back rather than letting the caller redo the rotations is deliberate: a
        second C rotated differently from B would break the relative-position
        property the rotary exists to provide, and that failure is silent -- it
        cost the 2-directional operator 12 points before it was found once already
        (see ``rotate_pairs`` in rope2d.py).
        """
        B_t, C_t, V_t, _delta, A_log, _lam, angles = self.projections(x)

        rotations = []
        if angles is not None:
            # Mamba-3's rotary, adapted to the collapse. The SSD form accumulates
            # angle*Delta along the scan, but Delta does not survive the NC
            # reduction (doc section 6.7.4), so the increment is unscaled: theta is
            # a plain cumsum, i.e. position advances one step per token rather
            # than data-dependently. Rotating B and C alike still makes the
            # C_i . B_j score depend on the relative angle, which is the point.
            theta = torch.cumsum(angles, dim=-2)
            rotations.append(lambda t: apply_cumulative_rope(t, theta))
        if self.rope is not None and pos is not None:
            rotations.append(lambda t: self.rope(t, pos))

        def rotate(t: Tensor) -> Tensor:
            for f in rotations:
                t = f(t)
            return t

        # VSSD §3.2: learn m directly. AttentionProjections returns
        # `A_log = -softplus(...)` (strictly negative). softplus(|A_log|)
        # gives a strictly-positive m of comparable dynamic range.
        m = self._mask_m(F.softplus(-A_log), attn_mask)         # (B, H, T)
        return rotate(B_t), rotate(C_t), V_t, m, rotate

    @staticmethod
    def _mask_m(m: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        """Zero masked tokens' weight so they never enter the global state."""
        if attn_mask is not None and attn_mask.ndim == 2:
            m = m * attn_mask.to(m.dtype).unsqueeze(1)          # (B, H, T)
        return m

    def _finish(self, y: Tensor) -> Tensor:
        Bsz, H, T, hd = y.shape
        y = y.transpose(1, 2).contiguous().view(Bsz, T, H * hd)
        return self.proj(self.post_norm(y))


def vssd_beta_gamma_forward(
    B: Tensor, C1: Tensor, C2: Tensor, V: Tensor, m1: Tensor, m2: Tensor
) -> Tensor:
    """Apply the VSSD-beta,gamma output formula, doc eq. (vssd-beta-gamma):

        Y = C1 ((B (*) m1)^T X) + C2 ((B (*) m2)^T X)

    Two independently-read global pools. ``B`` and ``V`` are shared; only the
    query and the per-token weight differ per pool.

    The independence of ``C2`` is the whole content of the operator, not a
    detail. With a shared query the two terms factor back together,

        C H1 + C H2 = C (H1 + H2) = C ((B (*) (m1 + m2))^T X),

    which is plain VSSD-gamma with a combined weight vector -- the second pool
    would be unable to express anything the first could not, since a shared C
    sums the rows away before the output is formed. Giving pool 2 its own query
    is what stops that factoring. ``tests/unit/test_vssd_beta_gamma.py`` asserts
    both halves: that sharing C collapses exactly, and that not sharing it does
    not.
    """
    return vssd_forward(B, C1, V, m1) + vssd_forward(B, C2, V, m2)


class Mamba3VSSDBetaGammaAttention(Mamba3VSSDAttention):
    """VSSD-beta,gamma self-attention (doc section 6.7.9, eq. vssd-beta-gamma).

    VSSD-gamma's mask collapses to one per-token scalar, which carries no |i-j|
    term at all. This variant adds a second, independently-read global pool,
    motivated by the beta band of Mamba-3's trapezoidal mask: the doc shows the
    causal L3 mask is exactly as low-rank off the diagonal as Mamba-2's, so beta's
    causal content costs nothing to recover, but that the *bidirectional* collapse
    needs two pools because the forward pass wants beta_{j+1} where the reverse
    wants beta_{j-1}, and a single query cannot keep them apart.

    Cost against VSSD-gamma: one extra C projection and one extra scalar
    projection, 2x the O(ND) state and 2x the einsum work. Still independent of T
    in memory, and still strictly cheaper than bidirectional SSD's O(T^2(N+D))
    once T is large.

    What this is not: a proof that pool 2 recovers Mamba-3's beta-band behaviour.
    m2 is a freely-learned per-token vector, motivated by -- not constrained to
    equal -- beta_{j+1}. It buys comparable *capacity* at 2x cost. Whether
    initialising or regularising it toward beta-like behaviour beats leaving it
    free is untested.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        H, N = self.num_heads, self.state_dim
        # Pool 2's own query and scalar weight. B and V stay shared, so this is
        # one linear of H*N + H outputs, mirroring the C and A rows of
        # AttentionProjections rather than duplicating the whole bundle.
        self.proj2 = nn.Linear(self.dim, H * N + H, bias=False)
        self.bc_norm_c2 = BCNorm(H, N)
        self.A2_bias = nn.Parameter(torch.zeros(H))

    def forward(
        self,
        x: Tensor,
        pos: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B_t, C1, V_t, m1, rotate = self._project(x, pos, attn_mask)

        Bsz, T, _ = x.shape
        H, N = self.num_heads, self.state_dim
        p2 = self.proj2(x)
        C2 = self.bc_norm_c2(p2[..., : H * N].reshape(Bsz, T, H, N).transpose(1, 2))
        # Rotate pool 2's query exactly as B was rotated, or its scores stop
        # depending on relative position only.
        C2 = rotate(C2)
        # m2 built exactly like m1 (doc: "a second A^(2)_t, built exactly like
        # eq. projA but with independent weights"), so the two pools differ only
        # in their weights, never in their parameterisation.
        A2_log = -F.softplus(p2[..., H * N :].transpose(1, 2) + self.A2_bias[None, :, None])
        m2 = self._mask_m(F.softplus(-A2_log), attn_mask)

        return self._finish(vssd_beta_gamma_forward(B_t, C1, C2, V_t, m1, m2))
