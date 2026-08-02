"""Per-token projections for Mamba-3 SSD attention.

Produces B, C (state-dim), V (head-dim), plus the scalar-per-token Δ, A, λ
used to build the decay mask L.

Layout matches multi-head attention conventions:
    input  x:   (B, T, D)
    B, C:       (B, H, T, N_state)
    V:          (B, H, T, head_dim)    (head_dim = D / H)
    Δ, A, λ:    (B, H, T)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class BCNorm(nn.Module):
    """RMSNorm applied independently to each head's B (or C) stream, with a
    learnable per-channel bias. Stabilizes the CBᵀ similarity matrix.
    """

    def __init__(self, num_heads: int, state_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads, state_dim))
        self.bias = nn.Parameter(torch.zeros(num_heads, state_dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, H, T, N)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x = x * rms
        return x * self.weight[None, :, None, :] + self.bias[None, :, None, :]


class AttentionProjections(nn.Module):
    """Linear projections producing (B, C, V, Δ, A, λ) from token features.

    - B and C are the "key" and "query" analogs in state-space form. Each has
      state_dim entries per head.
    - V is the value, of shape (head_dim) per head.
    - Δ = softplus(·) per-token scalar per-head.
    - A = −softplus(·) per-token scalar per-head (strictly negative).
    - λ = sigmoid(·) per-token scalar per-head (in [0, 1]).

    The value is passed through SiLU, matching Mamba-3's pre-gating.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        state_dim: int = 64,
        bias: bool = False,
        rope_angles: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.state_dim = state_dim
        # Mamba-3's complex-SSM rotary: a learned per-token angle increment per
        # state channel-pair, shared across heads, matching upstream Mamba3's
        # `num_rope_angles = split_tensor_size // 2`. Off by default so the
        # projection keeps its original width and existing checkpoints load.
        self.num_rope_angles = state_dim // 2 if rope_angles else 0

        # One big linear for efficiency; slice the output.
        #   B: H * state_dim
        #   C: H * state_dim
        #   V: H * head_dim = dim
        #   Δ, A, λ: H each (scalar per head per token)
        #   angles: num_rope_angles (shared across heads), only if enabled
        self.out_size = (
            num_heads * state_dim  # B
            + num_heads * state_dim  # C
            + dim  # V
            + num_heads  # Δ
            + num_heads  # A
            + num_heads  # λ
            + self.num_rope_angles
        )
        self.proj = nn.Linear(dim, self.out_size, bias=bias)

        if self.num_rope_angles:
            # Zero-init the angle rows so the rotary is the identity at step 0.
            # Two reasons. (1) Swapping the rotary into a pretrained model then
            # starts exactly at the un-rotated model, like the other zero-init
            # heads here. (2) From a random init the angle increments cumsum over
            # the sequence into huge rotations that scramble the C.B similarity,
            # and the kernel's fast cos_approx/sin_approx drift from exact trig
            # there -- reference-vs-kernel agreement falls from 0.98 to 0.79 at
            # T=128, and worsens with T.
            with torch.no_grad():
                self.proj.weight[-self.num_rope_angles :].zero_()
                if bias:
                    self.proj.bias[-self.num_rope_angles :].zero_()

        self.bc_norm_b = BCNorm(num_heads, state_dim)
        self.bc_norm_c = BCNorm(num_heads, state_dim)

        # Per-head bias for Δ, A, λ (scalar-per-head streams). Defaults to zero
        # so existing behaviour is unchanged. `warm_start_mamba3_from_qkv` sets
        # these so that at day 0 the SSD mask is approximately uniform (softmax-
        # like) instead of steeply decaying from random projection outputs.
        self.delta_bias = nn.Parameter(torch.zeros(num_heads))
        self.A_bias = nn.Parameter(torch.zeros(num_heads))
        self.lam_bias = nn.Parameter(torch.zeros(num_heads))

    def forward(self, x: Tensor):
        """
        Args:
            x: (B, T, D)

        Returns:
            B_t:    (B, H, T, N_state)
            C_t:    (B, H, T, N_state)
            V_t:    (B, H, T, head_dim)   (after SiLU)
            delta:  (B, H, T)              (> 0)
            A_log:  (B, H, T)              (< 0)
            lam:    (B, H, T)              (∈ [0, 1])
            angles: (B, H, T, N_state/2) rotary angle increments, or None when
                    `rope_angles=False`. Broadcast across heads, as upstream does.
        """
        B, T, D = x.shape
        H, N, hd = self.num_heads, self.state_dim, self.head_dim

        proj = self.proj(x)
        s1 = H * N
        s2 = H * N
        s3 = D
        B_raw = proj[..., :s1].reshape(B, T, H, N).transpose(1, 2)  # (B, H, T, N)
        C_raw = proj[..., s1 : s1 + s2].reshape(B, T, H, N).transpose(1, 2)
        V_raw = proj[..., s1 + s2 : s1 + s2 + s3].reshape(B, T, H, hd).transpose(1, 2)
        rest = proj[..., s1 + s2 + s3 :]
        delta_raw = rest[..., :H].transpose(1, 2)  # (B, H, T)
        A_raw = rest[..., H : 2 * H].transpose(1, 2)
        lam_raw = rest[..., 2 * H : 3 * H].transpose(1, 2)
        angles = None
        if self.num_rope_angles:
            angles = rest[..., 3 * H : 3 * H + self.num_rope_angles]  # (B, T, A)
            angles = angles.unsqueeze(1).expand(-1, H, -1, -1)  # (B, H, T, A)

        B_t = self.bc_norm_b(B_raw)
        C_t = self.bc_norm_c(C_raw)
        V_t = F.silu(V_raw)
        delta = F.softplus(delta_raw + self.delta_bias[None, :, None])
        A_log = -F.softplus(A_raw + self.A_bias[None, :, None])
        lam = torch.sigmoid(lam_raw + self.lam_bias[None, :, None])
        return B_t, C_t, V_t, delta, A_log, lam, angles
