"""Mamba-3 self-attention (SSD) as a drop-in replacement for softmax attention.

Output formula per head:
    Y_h = (L_h ⊙ (C_h · B_hᵀ)) · V_h     shape (T, head_dim)

Where L is the structured decay mask from mask.build_three_term_mask.

Bidirectional variant sums the forward and reversed SSD (paper eq. 552).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .mask import (
    build_three_term_mask,
    build_three_term_mask_rows,
    build_two_term_mask,
    build_two_term_mask_rows,
)
from .projections import AttentionProjections


def apply_cumulative_rope(t: Tensor, theta: Tensor) -> Tensor:
    """Rotate state-channel pairs of `t` by per-token cumulative angles `theta`.

    This is Mamba-3's complex-SSM rotary written out in PyTorch, and the
    reference-path twin of what the Triton kernel does internally from its
    `Angles` argument. Rotating both B and C by their own absolute angle makes
    the SSD score C_i . B_j depend on the *relative* rotation theta_i - theta_j.

    Args:
        t:     (B, H, T, N) state projection, N even.
        theta: (B, H, T, N/2) cumulative angle per channel pair.
    """
    # Interleaved pairs (t0,t1), (t2,t3), ... -- matching the kernel, which does
    # `tl.split(tl.reshape(k, [CHUNK_SIZE, HEADDIM_QK // 2, 2]))`. A half-split
    # convention silently disagrees with it rather than erroring.
    pairs = t.unflatten(-1, (t.shape[-1] // 2, 2))
    t0, t1 = pairs[..., 0], pairs[..., 1]
    cos, sin = torch.cos(theta), torch.sin(theta)
    return torch.stack([t0 * cos - t1 * sin, t0 * sin + t1 * cos], dim=-1).flatten(-2)


def ssd_forward(
    B: Tensor,
    C: Tensor,
    V: Tensor,
    L: Tensor,
    row_renorm: bool = False,
    eps: float = 1e-6,
) -> Tensor:
    """Apply SSD output formula given projections and mask.

    Args:
        B: (batch, H, T, N_state)
        C: (batch, H, T, N_state)
        V: (batch, H, T, head_dim)
        L: (batch, H, T, T) lower-triangular decay mask
        row_renorm: if True, divide each row of (L ⊙ (C·Bᵀ)) by the sum of the
            row's magnitudes before multiplying V. Gives softmax-like "row sums
            to 1" contract — needed when a downstream softmax-trained head
            (e.g. DA3 DualDPT) consumes these features.

    Returns:
        Y: (batch, H, T, head_dim)
    """
    sim = torch.matmul(C, B.transpose(-2, -1))
    weighted = sim * L
    if row_renorm:
        denom = weighted.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
        weighted = weighted / denom
    return torch.matmul(weighted, V)


def ssd_forward_chunked(
    B: Tensor,
    C: Tensor,
    V: Tensor,
    delta: Tensor,
    A_log: Tensor,
    lam: Tensor | None = None,
    three_term: bool = True,
    row_renorm: bool = False,
    chunk_size: int = 128,
    eps: float = 1e-6,
) -> Tensor:
    """Memory-efficient SSD forward: builds mask rows in chunks of `chunk_size`.

    Equivalent to `ssd_forward(B, C, V, build_*_mask(delta, A_log[, lam]), ...)`
    but never materializes the full (..., T, T) mask — peak memory per chunk is
    O(chunk_size * T) instead of O(T * T). Needed for high-resolution inference
    (R7 in PLAN §9) where T = (H/patch_size)² exceeds a few hundred tokens.

    Args mirror `ssd_forward`; `three_term` picks the Mamba-3 trapezoidal mask
    (requires `lam`) vs. the Mamba-2 two-term mask.
    """
    T = B.shape[-2]
    if chunk_size is None or chunk_size >= T:
        if three_term:
            assert lam is not None, "three_term=True requires lam"
            L = build_three_term_mask(delta, A_log, lam)
        else:
            L = build_two_term_mask(delta, A_log)
        return ssd_forward(B, C, V, L, row_renorm=row_renorm, eps=eps)

    out_chunks: list[Tensor] = []
    for q_start in range(0, T, chunk_size):
        q_end = min(q_start + chunk_size, T)
        if three_term:
            assert lam is not None, "three_term=True requires lam"
            L_rows = build_three_term_mask_rows(delta, A_log, lam, q_start, q_end)
        else:
            L_rows = build_two_term_mask_rows(delta, A_log, q_start, q_end)
        C_chunk = C[..., q_start:q_end, :]
        sim = torch.matmul(C_chunk, B.transpose(-2, -1))  # (..., chunk, T)
        weighted = sim * L_rows
        if row_renorm:
            denom = weighted.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
            weighted = weighted / denom
        out_chunks.append(torch.matmul(weighted, V))
    return torch.cat(out_chunks, dim=-2)


class Mamba3SelfAttention(nn.Module):
    """Bidirectional Mamba-3 self-attention.

    Args:
        dim:          token feature dimension D
        num_heads:    H (D must be divisible by H)
        state_dim:    N_state per head (default 64)
        bidirectional: if True, sum forward and reverse SSD
        three_term:   if True, use Mamba-3 trapezoidal mask; else Mamba-2 two-term
        rope:         optional module implementing forward(tokens, positions),
                      applied to B/C before the SSD (e.g. 2-D RoPE for images)
        out_proj:     if True, apply a final linear projection (like Attention.proj)
        rope_angles:  if True, enable Mamba-3's own complex-SSM rotary -- a learned
                      per-token angle increment per state channel-pair, which the
                      kernel accumulates as cumsum(Angles*DT). Independent of
                      `rope`: that one is an absolute encoding applied outside,
                      this one is relative and internal to the operator.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        state_dim: int = 64,
        bidirectional: bool = True,
        three_term: bool = True,
        rope: Optional[nn.Module] = None,
        out_proj: bool = True,
        proj_bias: bool = True,
        row_renorm: bool = True,
        post_norm: bool = True,
        chunk_size: Optional[int] = None,
        use_fused_kernel: bool = True,
        rope_angles: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.state_dim = state_dim
        self.bidirectional = bidirectional
        self.three_term = three_term
        self.rope = rope
        self.row_renorm = row_renorm
        self.chunk_size = chunk_size
        self.use_fused_kernel = use_fused_kernel

        self.projections = AttentionProjections(
            dim, num_heads, state_dim, rope_angles=rope_angles
        )
        # Per-head pre-tanh gate for the reverse SSD stream. Zero-init means
        # tanh(0)=0, so at init the layer behaves as forward-only (reverse noise
        # disabled). Training opens the gate as needed (R2 in PLAN §9).
        self.rev_gate = nn.Parameter(torch.zeros(num_heads)) if bidirectional else None
        self.post_norm = nn.LayerNorm(dim) if post_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias) if out_proj else nn.Identity()

    def _build_mask(self, delta: Tensor, A_log: Tensor, lam: Tensor) -> Tensor:
        if self.three_term:
            return build_three_term_mask(delta, A_log, lam)
        return build_two_term_mask(delta, A_log)

    def _one_direction(
        self,
        Bp: Tensor,
        Cp: Tensor,
        Vp: Tensor,
        delta: Tensor,
        A_log: Tensor,
        lam: Tensor,
        angles: Optional[Tensor] = None,
    ) -> Tensor:
        # The Triton kernel requires CUDA; fall back to the PyTorch path on
        # CPU so unit tests and ad-hoc CPU runs still work.
        if self.use_fused_kernel and Bp.is_cuda:
            return self._one_direction_kernel(Bp, Cp, Vp, delta, A_log, lam, angles)
        if angles is not None:
            # Reference-path equivalent of the kernel's internal rotary: it forms
            # Angles_Cumsum = cumsum(Angles * DT) and rotates B and C by it, so the
            # C_i . B_j score sees the *relative* rotation between the two tokens.
            theta = torch.cumsum(angles * delta.unsqueeze(-1), dim=-2)
            Bp = apply_cumulative_rope(Bp, theta)
            Cp = apply_cumulative_rope(Cp, theta)
        if self.chunk_size is not None and self.chunk_size < Bp.shape[-2]:
            return ssd_forward_chunked(
                Bp, Cp, Vp, delta, A_log, lam,
                three_term=self.three_term,
                row_renorm=self.row_renorm,
                chunk_size=self.chunk_size,
            )
        L = self._build_mask(delta, A_log, lam)
        return ssd_forward(Bp, Cp, Vp, L, row_renorm=self.row_renorm)

    def _one_direction_kernel(
        self,
        Bp: Tensor,
        Cp: Tensor,
        Vp: Tensor,
        delta: Tensor,
        A_log: Tensor,
        lam: Tensor,
        angles: Optional[Tensor] = None,
    ) -> Tensor:
        """Mamba-3 SISO Triton kernel path. State-spaces/mamba >= v2.3.1.

        The kernel signature uses (Q, K, V) attention naming; SSD-DA3 maps:
            our Cp (query proj)   → kernel Q   (B, T, H, state_dim)
            our Bp (key proj)     → kernel K   (B, T, H, state_dim)
            our Vp (values)       → kernel V   (B, T, H, head_dim)
            our delta * A_log     → kernel ADT (B, H, T)   — α_t = exp(δ·A_log_t)
            our delta             → kernel DT  (B, H, T)
            our lam (sigmoid)     → kernel Trap(B, H, T)   — λ_t ∈ [0, 1]

        RoPE: with `rope_angles=True` we hand the kernel the learned per-token
        angle increments and let it run Mamba-3's own complex-SSM rotary
        (Angles_Cumsum = cumsum(Angles·DT)). With `rope_angles=False` we pass
        zeros, making the rotation the identity, for callers that instead apply
        an external positional encoding to B/C before this point.

        `row_renorm` (softmax-like row normalisation) is **not** supported by
        the upstream kernel — it is a SSM-3D-specific design choice that
        injects a divide-by-row-magnitude after the SSD matmul. Caller must
        set `row_renorm=False` (or the renorm should be added as a
        post-kernel correction, not done here).
        """
        from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

        Bsz, H, T, headdim_v = Vp.shape
        state_dim = Bp.shape[-1]
        out_dtype = Vp.dtype

        # The Triton kernel uses fp32 accumulators in `tl.dot`, so its inputs
        # must be fp32 even when the surrounding model runs under bf16 autocast.
        # Upcast here, downcast the result before returning.
        Q = Cp.transpose(1, 2).contiguous().float()
        K = Bp.transpose(1, 2).contiguous().float()
        V = Vp.transpose(1, 2).contiguous().float()

        ADT = (delta * A_log).float()
        DT = delta.float()
        Trap = lam.float()

        Q_bias = torch.zeros(H, state_dim, dtype=torch.float32, device=Q.device)
        K_bias = torch.zeros(H, state_dim, dtype=torch.float32, device=K.device)
        # headdim_angles = state_dim // 2, the rotary's natural half-pair size;
        # also avoids the kernel's degenerate `headdim_angles=0` case.
        headdim_angles = state_dim // 2
        if angles is None:
            Angles = torch.zeros(
                Bsz, T, H, headdim_angles, dtype=torch.float32, device=Q.device,
            )
        else:  # (B, H, T, A) -> the kernel's (B, T, H, A)
            Angles = angles.permute(0, 2, 1, 3).contiguous().float()

        out = mamba3_siso_combined(
            Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
            chunk_size=self.chunk_size if self.chunk_size is not None else 64,
        )
        return out.transpose(1, 2).contiguous().to(out_dtype)

    def forward(
        self,
        x: Tensor,
        pos: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:         (B, T, D)
            pos:       (B, T, 2) integer 2D positions for RoPE, or None
            attn_mask: (B, T, T) additive/boolean mask OR (B, T); True/finite → keep.
                       When provided, zeros-out columns of L before the SSD
                       (so masked tokens don't contribute regardless of decay).

        Returns:
            y: (B, T, D)
        """
        Bp, Cp, Vp, delta, A_log, lam, angles = self.projections(x)

        if self.rope is not None and pos is not None:
            Bp = self.rope(Bp, pos)
            Cp = self.rope(Cp, pos)

        y = self._one_direction(Bp, Cp, Vp, delta, A_log, lam, angles)

        if self.bidirectional:
            # Reverse along T
            Bp_r = Bp.flip(dims=(-2,))
            Cp_r = Cp.flip(dims=(-2,))
            Vp_r = Vp.flip(dims=(-2,))
            delta_r = delta.flip(dims=(-1,))
            A_log_r = A_log.flip(dims=(-1,))
            lam_r = lam.flip(dims=(-1,))
            angles_r = None if angles is None else angles.flip(dims=(-2,))
            y_rev = self._one_direction(
                Bp_r, Cp_r, Vp_r, delta_r, A_log_r, lam_r, angles_r
            )
            y_rev = y_rev.flip(dims=(-2,))
            gate = torch.tanh(self.rev_gate)[None, :, None, None]  # (1, H, 1, 1)
            y = y + gate * y_rev

        if attn_mask is not None:
            # Token-zero-out semantics: if the row-i column-j is masked, remove
            # contribution of kv-token j from query-token i. For simplicity we
            # accept (B, T) that masks out whole kv-tokens.
            if attn_mask.ndim == 2:
                keep = attn_mask.to(y.dtype)  # (B, T)
                y = y * keep[:, None, :, None]

        # Merge heads: (B, H, T, head_dim) -> (B, T, D)
        Bsz, H, T, hd = y.shape
        y = y.transpose(1, 2).contiguous().view(Bsz, T, H * hd)
        y = self.post_norm(y)
        return self.proj(y)
