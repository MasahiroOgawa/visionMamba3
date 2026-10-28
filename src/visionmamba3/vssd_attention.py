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

import math

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
        # Diagnostic: scale the per-token angle increment. theta is a plain cumsum here,
        # so its spread across the sequence grows linearly with T; at T=1025 that is
        # ~160 turns, and the pooled state sum_j m_j R(theta_j) B_j decoheres. Setting
        # this to 65/1025 gives T=1025 the same total angular spread T=65 had, which is
        # the test of whether spread -- not the rotary itself -- is what breaks training.
        # Wrapping theta would be a no-op: cos/sin are 2pi-periodic, so only the spread
        # matters, not the absolute value.
        rope_angle_scale: float = 1.0,
        # Fixed total angular spread, in whole turns, instead of a learned cumulative
        # angle: theta_j = j * 2*pi*n/T, so the sequence spans exactly `n` turns whatever
        # T is. This removes the coupling that makes the learned form fail at long T --
        # there, neighbour resolution is the increment and total spread is T times it, one
        # parameter for two quantities whose ratio is fixed at T. Pinning the spread costs
        # the data-dependence of the angle; the increment becomes purely positional.
        rope_turns: float | None = None,
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
        self.rope_angle_scale = float(rope_angle_scale)
        self.rope_turns = None if rope_turns is None else float(rope_turns)

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
            if self.rope_turns is not None:
                # Positional, not learned: one step per token, sized so the whole
                # sequence covers `rope_turns` turns. `angles` is still projected (the
                # width is baked into the checkpoint) but only its shape is used.
                T = angles.shape[-2]
                step = 2.0 * math.pi * self.rope_turns / max(T, 1)
                ramp = torch.arange(T, device=angles.device, dtype=angles.dtype) * step
                theta = ramp[None, None, :, None].expand_as(angles)
            else:
                theta = torch.cumsum(angles, dim=-2) * self.rope_angle_scale
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
        return self._finish(vssd_beta_gamma_forward(*self._streams(x, pos, attn_mask)))

    def _streams(self, x: Tensor, pos: Optional[Tensor], attn_mask: Optional[Tensor]):
        """``(B, C1, C2, V, m1, m2)`` of eq. (vssd-beta-gamma)."""
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
        return B_t, C1, C2, V_t, m1, m2

    # -- frame-causal streaming form -------------------------------------------

    def enable_frame_stream(self, ref_frames: int) -> None:
        """Add the per-head frame decay used by :meth:`forward_frames`.

        ``ref_frames`` is the clip length the weights were trained on. The read is
        scaled by ``ref_frames / c_t`` (``c_t`` the decayed frame count), so at the
        last frame of such a clip, with decay 1, it equals the windowed pool exactly,
        and on a long stream it stays at that clip's scale instead of growing.
        """
        if ref_frames < 1:
            raise ValueError(f"ref_frames must be >= 1, got {ref_frames}")
        self.frame_ref = int(ref_frames)
        # gamma = exp(-softplus(p)); p = -10 gives gamma = 1 - 4.5e-5 per frame, so a
        # warm-started stream matches the windowed operator to that relative error while
        # gamma stays learnable (softplus has no flat region at p = -10 under Adam).
        self.frame_decay = nn.Parameter(torch.full((self.num_heads,), -10.0))

    def frame_gamma(self) -> Tensor:
        return torch.exp(-F.softplus(self.frame_decay))

    def _frame_rotation(self, row_step: int, device, dtype) -> Optional[tuple[Tensor, Tensor]]:
        """cos/sin of the per-frame state rotation, or None without 2-D RoPE.

        The windowed mixer separates frame v by a RoPE row offset ``y + v*row_step``.
        RoPE scores depend only on angle differences, so the same separation is a
        rotation of the stored state by ``-row_step`` rows per elapsed frame, on the
        row-encoded planes (first half of N) only. Computed from the angle, not from
        RoPE's position table, so no frame index ever leaves the table.
        """
        if self.rope is None:
            return None
        n_pairs = self.state_dim // 4
        exps = torch.arange(n_pairs, device=device, dtype=torch.float64) / n_pairs
        angle = -float(row_step) / (float(self.rope.base_frequency) ** exps)
        return angle.cos().to(dtype), angle.sin().to(dtype)

    @staticmethod
    def _rotate_state(S: Tensor, rot: Optional[tuple[Tensor, Tensor]]) -> Tensor:
        """Rotate the row-encoded state planes of ``S`` (b, H, N, d)."""
        if rot is None:
            return S
        cos, sin = (t[:, None] for t in rot)
        n_half = S.shape[2] // 2
        pairs = S[:, :, :n_half].unflatten(2, (n_half // 2, 2))
        s0, s1 = pairs[..., 0, :], pairs[..., 1, :]
        rowpart = torch.stack([s0 * cos - s1 * sin, s0 * sin + s1 * cos], dim=3).flatten(2, 3)
        return torch.cat([rowpart, S[:, :, n_half:]], dim=2)

    def forward_frames(
        self,
        x: Tensor,
        pos: Optional[Tensor],
        n_frames: int,
        row_step: int,
        state: Optional[tuple[Tensor, Tensor, Tensor]] = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        """Frame-causal eq. (vssd-beta-gamma): each frame reads itself and the past.

        Args:
            x: (b, n_frames*T, D), frame-major.
            pos: (b, n_frames*T, 2) WITHIN-frame (y, x) positions -- no frame offset;
                the offset is the state rotation (:meth:`_frame_rotation`).
            n_frames: frames in ``x``.
            row_step: token rows per frame (the windowed offset's ``gh``).
            state: ``(S1, S2, c)`` from the previous call, or None to start a stream.

        Per frame t, with P^(k)_t the frame's own pool (B (*) m^(k))^T V:

            S^(k)_t = gamma R S^(k)_{t-1} + P^(k)_t,   c_t = gamma c_{t-1} + 1,
            Y_t     = (ref / c_t) * sum_k C^(k)_t S^(k)_t.

        Calling it once on n frames or n times on one frame gives the same result.
        """
        if not hasattr(self, "frame_decay"):
            raise RuntimeError("forward_frames needs enable_frame_stream() first")
        B_t, C1, C2, V_t, m1, m2 = self._streams(x, pos, None)
        Bsz, H, L, N = B_t.shape
        if L % n_frames:
            raise ValueError(f"{L} tokens do not split into {n_frames} frames")

        def frames(t: Tensor) -> Tensor:
            return t.unflatten(2, (n_frames, L // n_frames))

        Bf, Vf, C1f, C2f = frames(B_t), frames(V_t), frames(C1), frames(C2)
        P1 = torch.einsum("bhvtn,bhvtd->bhvnd", Bf * frames(m1).unsqueeze(-1), Vf)
        P2 = torch.einsum("bhvtn,bhvtd->bhvnd", Bf * frames(m2).unsqueeze(-1), Vf)
        if state is None:
            S1 = torch.zeros_like(P1[:, :, 0])
            S2 = torch.zeros_like(P2[:, :, 0])
            c = torch.zeros(Bsz, H, device=x.device, dtype=P1.dtype)
        else:
            S1, S2, c = state
        gamma = self.frame_gamma().to(P1.dtype)
        rot = self._frame_rotation(row_step, x.device, P1.dtype)
        ys = []
        for v in range(n_frames):
            S1 = gamma[None, :, None, None] * self._rotate_state(S1, rot) + P1[:, :, v]
            S2 = gamma[None, :, None, None] * self._rotate_state(S2, rot) + P2[:, :, v]
            c = gamma[None] * c + 1
            y = (torch.einsum("bhtn,bhnd->bhtd", C1f[:, :, v], S1)
                 + torch.einsum("bhtn,bhnd->bhtd", C2f[:, :, v], S2))
            ys.append(y * (self.frame_ref / c)[..., None, None])
        return self._finish(torch.cat(ys, dim=2)), (S1, S2, c)
