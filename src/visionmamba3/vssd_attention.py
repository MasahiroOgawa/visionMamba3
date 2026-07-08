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

from .projections import AttentionProjections


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

        self.projections = AttentionProjections(dim, num_heads, state_dim)
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
        B_t, C_t, V_t, _delta, A_raw, _lam = self.projections(x)

        if self.rope is not None and pos is not None:
            B_t = self.rope(B_t, pos)
            C_t = self.rope(C_t, pos)

        # VSSD §3.2: learn m directly. The existing AttentionProjections returns
        # `A_log = -softplus(...)` (strictly negative). softplus(|A_log|)
        # gives a strictly-positive m of comparable dynamic range.
        m = F.softplus(-A_raw)                                  # (B, H, T)

        if attn_mask is not None and attn_mask.ndim == 2:
            keep = attn_mask.to(m.dtype)                        # (B, T)
            m = m * keep.unsqueeze(1)                           # (B, H, T)

        y = vssd_forward(B_t, C_t, V_t, m)                      # (B, H, T, head_dim)

        Bsz, H, T, hd = y.shape
        y = y.transpose(1, 2).contiguous().view(Bsz, T, H * hd)
        y = self.post_norm(y)
        return self.proj(y)
