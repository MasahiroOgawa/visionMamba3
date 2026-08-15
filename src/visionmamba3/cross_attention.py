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

from .mask import build_cross_mask, build_cross_scale
from .projections import AttentionProjections


COLLAPSE = "collapse"
TOKEN_LEVEL_TEST_ONLY = "OnlyForCodingCorrectnessTestTokenLevelPathOrderTSquare"
class Mamba3CrossAttention(nn.Module):
    """Cross-attention where query and kv come from different token streams.

    Args:
        dim_q, dim_kv:  feature dims of query and kv token streams
        num_heads:      heads; both dims must be divisible by H
        state_dim:      N_state per head
        variant:        COLLAPSE (default) pools the keys into an (N, head_dim) state and reads
                        it with C -- eq. (4) as written, linear in T. TOKEN_LEVEL_TEST_ONLY
                        instead materialises the full T_q x T_kv similarity and multiplies it by
                        the expanded mask; it computes the same function at O(T^2) cost and
                        exists only to check the collapse against a direct transcription of the
                        formula.
        out_proj:       apply output linear if True
        two_pool:       add a second, independently-read pool (VSSD-beta,gamma). Variant B's
                        mask is rank-1 across the query axis -- every query row of L_cross is
                        identical -- so one pool applies the same weighting over the kv
                        sequence to every query. A second pool with its OWN query projection
                        and its own per-token vector lifts that restriction, which is the
                        construction of the paper's VSSD-beta,gamma. Its contribution is
                        scaled by a parameter initialised to zero, so a freshly-built
                        two-pool layer is numerically identical to a one-pool layer and a
                        fine-tune starts exactly at the one-pool result rather than at a
                        perturbation of it.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int = 8,
        state_dim: int = 64,
        variant: str = COLLAPSE,
        out_proj: bool = True,
        proj_bias: bool = True,
        bidirectional_mask: bool = False,
        two_pool: bool = False,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        assert variant in (COLLAPSE, TOKEN_LEVEL_TEST_ONLY), (
            f"variant must be {COLLAPSE!r} or {TOKEN_LEVEL_TEST_ONLY!r}, got {variant!r}")

        assert dim_q % num_heads == 0 and dim_kv % num_heads == 0

        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.state_dim = state_dim
        self.variant = variant
        self.bidirectional_mask = bidirectional_mask
        self.head_dim_kv = dim_kv // num_heads

        self.two_pool = two_pool
        # Variant B materialises (B, H, T_q, T_kv) tensors -- L, sim, and with two_pool a second
        # sim -- so peak memory grows with batch x T^2. At TAPVid-3D's adt shape (849 tracks,
        # T=300) one of them is already 2.4 GB, and the second pool's 1.58x pushed a 12 GB card
        # over. Batch elements are independent here (BCNorm is per-head RMSNorm; nothing reduces
        # across dim 0), so slicing the batch is exact, not an approximation.
        self.chunk_size = chunk_size
        self.q_proj = AttentionProjections(dim_q, num_heads, state_dim)
        self.kv_proj = AttentionProjections(dim_kv, num_heads, state_dim)
        if two_pool:
            # C^(2): the second pool's own query projection.
            self.q_proj2 = AttentionProjections(dim_q, num_heads, state_dim)
            # m^(2): a second per-kv-token, per-head vector, freely learned from the token
            # rather than tied to A_j. softplus keeps it positive without bounding it above.
            self.m2 = nn.Linear(dim_kv, num_heads, bias=True)
            nn.init.zeros_(self.m2.weight)
            nn.init.zeros_(self.m2.bias)
            # Zero gate: the second pool contributes nothing until trained.
            self.pool2_gate = nn.Parameter(torch.zeros(1))

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
        if self.chunk_size and q_tokens.shape[0] > self.chunk_size:
            outs, attns = [], []
            for i in range(0, q_tokens.shape[0], self.chunk_size):
                sl = slice(i, i + self.chunk_size)
                r = self.forward(
                    q_tokens[sl], kv_tokens[sl],
                    None if q_pos is None else q_pos[sl],
                    None if kv_pos is None else kv_pos[sl],
                    rope, return_attn)
                if return_attn:
                    outs.append(r[0]); attns.append(r[1])
                else:
                    outs.append(r)
            y = torch.cat(outs, dim=0)
            return (y, torch.cat(attns, dim=0)) if return_attn else y

        # Queries: we only use C^q and (for variant B) the full decay isn't
        # applied on the query side.
        _Bq_unused, Cq, _Vq_unused, _dq, _Aq, _lq, _angq = self.q_proj(q_tokens)
        Bkv, Ckv_unused, Vkv, dkv, Akv, _lkv, _angkv = self.kv_proj(kv_tokens)

        if rope is not None and q_pos is not None:
            Cq = rope(Cq, q_pos)
        if rope is not None and kv_pos is not None:
            Bkv = rope(Bkv, kv_pos)

        if self.variant == TOKEN_LEVEL_TEST_ONLY:
            Cq2 = None
            if self.two_pool:
                _b2, Cq2, _v2, _d2, _a2, _l2, _ang2 = self.q_proj2(q_tokens)
                if rope is not None and q_pos is not None:
                    Cq2 = rope(Cq2, q_pos)
            return self._variant_b(Cq, Bkv, Vkv, dkv, Akv, return_attn,
                                   Cq2=Cq2, kv_tokens=kv_tokens)
        Cq2 = None
        if self.two_pool:
            _b2, Cq2, _v2, _d2, _a2, _l2, _ang2 = self.q_proj2(q_tokens)
            if rope is not None and q_pos is not None:
                Cq2 = rope(Cq2, q_pos)
        return self._variant_a(Cq, Bkv, Vkv, dkv, Akv, return_attn,
                               Cq2=Cq2, kv_tokens=kv_tokens)

    def _variant_b(
        self,
        Cq: Tensor,
        Bkv: Tensor,
        Vkv: Tensor,
        delta_kv: Tensor,
        A_log_kv: Tensor,
        return_attn: bool,
        Cq2: Optional[Tensor] = None,
        kv_tokens: Optional[Tensor] = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        T_q = Cq.shape[-2]
        L = build_cross_mask(delta_kv, A_log_kv, T_q, bidirectional=self.bidirectional_mask)  # (B, H, T_q, T_kv)
        sim = torch.matmul(Cq, Bkv.transpose(-2, -1))  # (B, H, T_q, T_kv)
        weighted = sim * L
        y = torch.matmul(weighted, Vkv)  # (B, H, T_q, head_dim_kv)

        if Cq2 is not None and kv_tokens is not None:
            # Second pool: its own query projection against the same keys, weighted by its own
            # per-token vector. Broadcast over the query axis exactly as L is, so this pool is
            # also rank-1 on its own -- the escape from rank-1 comes from summing two pools
            # read by different query projections, not from either one alone.
            m2 = nn.functional.softplus(self.m2(kv_tokens))            # (B, T_kv, H)
            m2 = m2.permute(0, 2, 1).unsqueeze(-2)                     # (B, H, 1, T_kv)
            sim2 = torch.matmul(Cq2, Bkv.transpose(-2, -1))            # (B, H, T_q, T_kv)
            y = y + self.pool2_gate * torch.matmul(sim2 * m2, Vkv)

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
        Cq2: Optional[Tensor] = None,
        kv_tokens: Optional[Tensor] = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        # The same vector the token-level path expands into a matrix -- one source of truth, so
        # the two paths cannot drift apart, and bidirectional is just the symmetric m.
        scale = build_cross_scale(delta_kv, A_log_kv, bidirectional=self.bidirectional_mask)

        # h_ref = Σ_j scale_j · B_j · V_jᵀ   ∈  (B, H, N, head_dim_kv)
        B_scaled = Bkv * scale.unsqueeze(-1)  # (B, H, T_kv, N)
        # einsum: "b h t n, b h t d -> b h n d"
        h_ref = torch.einsum("bhtn,bhtd->bhnd", B_scaled, Vkv)

        # y_i = Cq_i · h_ref   ∈ (B, H, T_q, head_dim_kv)
        y = torch.einsum("bhqn,bhnd->bhqd", Cq, h_ref)

        if Cq2 is not None and kv_tokens is not None:
            # The second pool in the same collapsed form: its own per-token vector m^(2) pools the
            # same keys into a second (N, head_dim) state, read by its own query projection. Two
            # pooled states, each O(ND) and independent of T -- which is the whole point of the
            # construction, and what the token-level path throws away by materialising T_q x T_kv.
            m2 = nn.functional.softplus(self.m2(kv_tokens))        # (B, T_kv, H)
            m2 = m2.permute(0, 2, 1)                               # (B, H, T_kv)
            h2 = torch.einsum("bhtn,bhtd->bhnd", Bkv * m2.unsqueeze(-1), Vkv)
            y = y + self.pool2_gate * torch.einsum("bhqn,bhnd->bhqd", Cq2, h2)

        Bsz, H, Tq, hd = y.shape
        y = y.transpose(1, 2).contiguous().view(Bsz, Tq, H * hd)
        y = self.out(y)
        if return_attn:
            return y, scale  # scale is the per-kv weighting used
        return y
