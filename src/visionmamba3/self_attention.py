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
from .rope2d import rotate_pairs


def apply_cumulative_rope(t: Tensor, theta: Tensor) -> Tensor:
    """Mamba-3's complex-SSM rotary in PyTorch: the reference-path twin of what the
    Triton kernel does internally from its `Angles` argument.

    Rotating both B and C by their own absolute cumulative angle makes the SSD score
    C_i . B_j depend on the *relative* rotation theta_i - theta_j. The pair layout
    comes from `rotate_pairs`, shared with 2-D RoPE so the two rotaries cannot drift
    into disjoint planes again -- see that function for what happens when they do.

    Args:
        t:     (B, H, T, N) state projection, N even.
        theta: (B, H, T, N/2) cumulative angle per channel pair.
    """
    return rotate_pairs(t, torch.cos(theta), torch.sin(theta))


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
        num_directions: 1 (causal), 2 (row-major forward+reverse, the default) or
                      4 (adds column-major forward+reverse). A raster scan makes
                      vertical neighbours T apart in scan order while horizontal
                      ones are adjacent; the column-major pair restores the other
                      axis. 4 requires `grid` at call time.
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
        num_directions: int = 2,
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
        if num_directions not in (1, 2, 4):
            raise ValueError(f"num_directions must be 1, 2 or 4, got {num_directions}")
        if num_directions == 4 and not bidirectional:
            raise ValueError("num_directions=4 implies bidirectional=True")
        self.num_directions = num_directions

        self.projections = AttentionProjections(
            dim, num_heads, state_dim, rope_angles=rope_angles
        )
        # Per-head pre-tanh gate for the reverse SSD stream. Zero-init means
        # tanh(0)=0, so at init the layer behaves as forward-only (reverse noise
        # disabled). Training opens the gate as needed (R2 in PLAN §9).
        self.rev_gate = nn.Parameter(torch.zeros(num_heads)) if bidirectional else None
        # Column-major forward/backward gates, same zero-init contract as
        # rev_gate: at init the layer is forward-row-major only, and training
        # opens each extra direction as it earns it. Kept as a separate
        # parameter (rather than widening rev_gate) so 2-directional
        # checkpoints still load.
        self.col_gates = (
            nn.Parameter(torch.zeros(2, num_heads)) if num_directions == 4 else None
        )
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

    @staticmethod
    def _column_major_index(grid, T: int, device):
        """Permutation taking row-major tokens to column-major order, and back.

        Any leading non-grid tokens (a CLS token, registers) are left in place:
        they have no (row, col) so they cannot be re-ordered, and they must stay
        at the same index for the residual stream to line up.
        """
        h, w = grid
        n_prefix = T - h * w
        if n_prefix < 0:
            raise ValueError(f"grid {grid} needs {h * w} tokens but T={T}")
        idx = torch.arange(h * w, device=device).view(h, w).t().reshape(-1) + n_prefix
        if n_prefix:
            idx = torch.cat([torch.arange(n_prefix, device=device), idx])
        return idx, torch.argsort(idx)

    def _scan(self, streams, perm=None, inv=None, flip: bool = False) -> Tensor:
        """One directional SSD pass, returned in the original token order.

        `perm` re-orders tokens before the scan (e.g. into column-major) and
        `inv` undoes it after; `flip` runs the scan backwards. Every per-token
        stream has to move together -- reordering B/C/V but not delta/A/lam would
        silently pair each token's projections with another token's decay.
        """
        Bp, Cp, Vp, delta, A_log, lam, angles = streams
        if perm is not None:
            Bp, Cp, Vp = Bp[..., perm, :], Cp[..., perm, :], Vp[..., perm, :]
            delta, A_log, lam = delta[..., perm], A_log[..., perm], lam[..., perm]
            angles = None if angles is None else angles[..., perm, :]
        if flip:
            Bp, Cp, Vp = Bp.flip(-2), Cp.flip(-2), Vp.flip(-2)
            delta, A_log, lam = delta.flip(-1), A_log.flip(-1), lam.flip(-1)
            angles = None if angles is None else angles.flip(-2)
        y = self._one_direction(Bp, Cp, Vp, delta, A_log, lam, angles)
        if flip:
            y = y.flip(-2)
        if inv is not None:
            y = y[..., inv, :]
        return y

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
        grid: Optional[tuple[int, int]] = None,
    ) -> Tensor:
        """
        Args:
            x:         (B, T, D)
            pos:       (B, T, 2) integer 2D positions for RoPE, or None
            grid:      (H, W) token grid, required when num_directions=4 so the
                       column-major scans know the 2-D layout. Leading non-grid
                       tokens (CLS) are allowed and stay in place.
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

        streams = (Bp, Cp, Vp, delta, A_log, lam, angles)
        y = self._scan(streams)

        if self.bidirectional:
            gate = torch.tanh(self.rev_gate)[None, :, None, None]  # (1, H, 1, 1)
            y = y + gate * self._scan(streams, flip=True)

        if self.num_directions == 4:
            if grid is None:
                raise ValueError(
                    "num_directions=4 needs the token grid: pass grid=(H, W). A "
                    "column-major scan is undefined on a bare token sequence."
                )
            perm, inv = self._column_major_index(grid, T=Bp.shape[-2], device=Bp.device)
            for k in range(2):
                gate = torch.tanh(self.col_gates[k])[None, :, None, None]
                y = y + gate * self._scan(streams, perm=perm, inv=inv, flip=bool(k))

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
