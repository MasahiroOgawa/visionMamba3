"""Standalone 2D Rotary Position Embedding used for testing Mamba-3 attention
in isolation. Functionally equivalent to DA3's RotaryPositionEmbedding2D so
we can pass either interchangeably in the adapter layer.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RoPE2D(nn.Module):
    def __init__(self, base_frequency: float = 100.0) -> None:
        super().__init__()
        self.base_frequency = base_frequency
        self._cache: dict[tuple, tuple[Tensor, Tensor]] = {}

    def _freqs(self, dim: int, max_pos: int, device, dtype) -> tuple[Tensor, Tensor]:
        key = (dim, max_pos, device, dtype)
        if key not in self._cache:
            exps = torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
            inv_freq = 1.0 / (self.base_frequency**exps)
            pos = torch.arange(max_pos, device=device, dtype=torch.float32)
            angles = torch.einsum("i,j->ij", pos, inv_freq)
            angles = torch.cat((angles, angles), dim=-1).to(dtype)
            self._cache[key] = (angles.cos(), angles.sin())
        return self._cache[key]

    @staticmethod
    def _rotate(x: Tensor) -> Tensor:
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, tokens: Tensor, positions: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # positions: (B, T) long; tokens: (B, H, T, d)
        cos_e = torch.nn.functional.embedding(positions, cos)[:, None]
        sin_e = torch.nn.functional.embedding(positions, sin)[:, None]
        return tokens * cos_e + self._rotate(tokens) * sin_e

    def forward(self, tokens: Tensor, positions: Tensor) -> Tensor:
        """Apply 2D RoPE.

        Args:
            tokens:    (B, H, T, d) with d divisible by 4
            positions: (B, T, 2)  integer y,x coords

        Returns:
            tokens of same shape with RoPE applied.
        """
        assert tokens.size(-1) % 4 == 0, "feature dim must be divisible by 4"
        assert positions.ndim == 3 and positions.size(-1) == 2

        feat = tokens.size(-1) // 2
        max_pos = int(positions.max().item()) + 1
        cos, sin = self._freqs(feat, max_pos, tokens.device, tokens.dtype)

        y, x = tokens.chunk(2, dim=-1)
        y = self._apply_rope(y, positions[..., 0].long(), cos, sin)
        x = self._apply_rope(x, positions[..., 1].long(), cos, sin)
        return torch.cat((y, x), dim=-1)
