"""Mamba-3 cross-attention.

Two variants from the paper:

  Variant A (state-compressed):
      h_ref = Σ_j γ^{kv}_j · (∏_{k=j+1..T_kv} α^{kv}_k) · B^{kv}_j · V^{kv}_jᵀ
            ∈ ℝ^{N × head_dim}
      y_i   = C^q_iᵀ · h_ref
      cost  O((T_q + T_kv) · N · D)

  Variant B (token-level):
      Y = (L_cross ⊙ (C^q · B^{kv}ᵀ)) · V^{kv}
      L_cross[i, j] = γ^{kv}_j · ∏_{k=j+1..T_kv} α^{kv}_k    (broadcast over i)
      cost O(T_q · T_kv · (N + D))

Variant B is useful for visualizing cross-view attention maps; variant A is
memory-efficient when T_kv ≫ T_q.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .mask import build_cross_mask
from .projections import AttentionProjections


class Mamba3CrossAttention(nn.Module):
    """Cross-attention where query and kv come from different token streams.

    Args:
        dim_q, dim_kv:  feature dims of query and kv token streams
        num_heads:      heads; both dims must be divisible by H
        state_dim:      N_state per head
        variant:        'A' (state-compressed) or 'B' (token-level / default)
        out_proj:       apply output linear if True
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int = 8,
        state_dim: int = 64,
        variant: str = "B",
        out_proj: bool = True,
        proj_bias: bool = True,
        bidirectional_mask: bool = False,
    ) -> None:
        super().__init__()
        assert variant in ("A", "B")
        assert dim_q % num_heads == 0 and dim_kv % num_heads == 0

        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.state_dim = state_dim
        self.variant = variant
        self.bidirectional_mask = bidirectional_mask
        self.head_dim_kv = dim_kv // num_heads

        self.q_proj = AttentionProjections(dim_q, num_heads, state_dim)
        self.kv_proj = AttentionProjections(dim_kv, num_heads, state_dim)

        # Output head_dim matches kv-side head_dim; final linear maps to dim_q.
        self.out = nn.Linear(dim_kv, dim_q, bias=proj_bias) if out_proj else nn.Identity()

    def forward(
        self,
        q_tokens: Tensor,
        kv_tokens: Tensor,
        q_pos: Optional[Tensor] = None,
        kv_pos: Optional[Tensor] = None,
        rope: Optional[nn.Module] = None,
        return_attn: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args:
            q_tokens:  (B, T_q, dim_q)
            kv_tokens: (B, T_kv, dim_kv)
            q_pos, kv_pos: optional 2D position tensors for RoPE
            rope: shared RoPE module applied to C^q and B^{kv} if positions given
            return_attn: if True, return (y, attn) where attn is the effective
                         similarity matrix (for visualization). Only meaningful
                         for variant B.

        Returns:
            y: (B, T_q, dim_q) and optionally the attention map.
        """
        # Queries: we only use C^q and (for variant B) the full decay isn't
        # applied on the query side.
        _Bq_unused, Cq, _Vq_unused, _dq, _Aq, _lq = self.q_proj(q_tokens)
        Bkv, Ckv_unused, Vkv, dkv, Akv, _lkv = self.kv_proj(kv_tokens)

        if rope is not None and q_pos is not None:
            Cq = rope(Cq, q_pos)
        if rope is not None and kv_pos is not None:
            Bkv = rope(Bkv, kv_pos)

        if self.variant == "B":
            return self._variant_b(Cq, Bkv, Vkv, dkv, Akv, return_attn)
        return self._variant_a(Cq, Bkv, Vkv, dkv, Akv, return_attn)

    def _variant_b(
        self,
        Cq: Tensor,
        Bkv: Tensor,
        Vkv: Tensor,
        delta_kv: Tensor,
        A_log_kv: Tensor,
        return_attn: bool,
    ) -> Tensor | tuple[Tensor, Tensor]:
        T_q = Cq.shape[-2]
        L = build_cross_mask(delta_kv, A_log_kv, T_q, bidirectional=self.bidirectional_mask)  # (B, H, T_q, T_kv)
        sim = torch.matmul(Cq, Bkv.transpose(-2, -1))  # (B, H, T_q, T_kv)
        weighted = sim * L
        y = torch.matmul(weighted, Vkv)  # (B, H, T_q, head_dim_kv)

        Bsz, H, Tq, hd = y.shape
        y = y.transpose(1, 2).contiguous().view(Bsz, Tq, H * hd)
        y = self.out(y)
        if return_attn:
            return y, weighted
        return y

    def _variant_a(
        self,
        Cq: Tensor,
        Bkv: Tensor,
        Vkv: Tensor,
        delta_kv: Tensor,
        A_log_kv: Tensor,
        return_attn: bool,
    ) -> Tensor | tuple[Tensor, Tensor]:
        # Build the per-kv-token scale vector
        log_alpha = delta_kv * A_log_kv  # (B, H, T_kv)
        log_gamma = torch.log(delta_kv.clamp_min(1e-20))
        S = torch.cumsum(log_alpha, dim=-1)
        S_total = S[..., -1:]
        log_col = log_gamma + (S_total - S)
        scale = log_col.exp()  # (B, H, T_kv)

        # h_ref = Σ_j scale_j · B_j · V_jᵀ   ∈  (B, H, N, head_dim_kv)
        B_scaled = Bkv * scale.unsqueeze(-1)  # (B, H, T_kv, N)
        # einsum: "b h t n, b h t d -> b h n d"
        h_ref = torch.einsum("bhtn,bhtd->bhnd", B_scaled, Vkv)

        # y_i = Cq_i · h_ref   ∈ (B, H, T_q, head_dim_kv)
        y = torch.einsum("bhqn,bhnd->bhqd", Cq, h_ref)

        Bsz, H, Tq, hd = y.shape
        y = y.transpose(1, 2).contiguous().view(Bsz, Tq, H * hd)
        y = self.out(y)
        if return_attn:
            return y, scale  # scale is the per-kv weighting used
        return y
