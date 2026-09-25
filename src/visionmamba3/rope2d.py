"""Standalone 2D Rotary Position Embedding used for testing Mamba-3 attention
in isolation. Functionally equivalent to DA3's RotaryPositionEmbedding2D so
we can pass either interchangeably in the adapter layer.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def rotate_pairs(t: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate *adjacent* channel pairs (t0,t1), (t2,t3), ... of `t` by cos/sin.

    The single implementation of this project's rotary convention. Every rotary --
    this file's 2-D RoPE and Mamba-3's own complex rotary in `self_attention` --
    must go through it, because a rotation is a rotation *in a plane* and RoPE's
    whole trick, that the C.B score depends only on the angle *difference* between
    two tokens, holds only when every rotary rotates in the *same* planes.

    That is not hypothetical. 2-D RoPE used to half-split instead, pairing channel
    k with k + d/4, i.e. disjoint planes from the kernel's
    `tl.split(tl.reshape(k, [T, N // 2, 2]))`. Each encoding was self-consistent
    alone, so single-encoding runs looked fine, but with both enabled every channel
    was spun about two different axes; those do not compose into one angle, so
    absolute position leaked into the score. Measured as departure from Toeplitz on
    constant content: 1.3e-2 mismatched vs 2.2e-7 matched, and it cost the
    2-directional operator 12 points on CIFAR-10 at T=65.
    """
    pairs = t.unflatten(-1, (t.shape[-1] // 2, 2))
    t0, t1 = pairs[..., 0], pairs[..., 1]
    return torch.stack([t0 * cos - t1 * sin, t0 * sin + t1 * cos], dim=-1).flatten(-2)


class RoPE2D(nn.Module):
    def __init__(self, base_frequency: float = 100.0) -> None:
        super().__init__()
        self.base_frequency = base_frequency
        self._cache: dict[tuple, tuple[Tensor, Tensor]] = {}

    def _freqs(self, n_pairs: int, max_pos: int, device, dtype) -> tuple[Tensor, Tensor]:
        """cos/sin of shape (max_pos, n_pairs) -- one angle per rotation plane."""
        key = (n_pairs, max_pos, device, dtype)
        if key not in self._cache:
            exps = torch.arange(n_pairs, device=device, dtype=torch.float32) / n_pairs
            inv_freq = 1.0 / (self.base_frequency**exps)
            pos = torch.arange(max_pos, device=device, dtype=torch.float32)
            angles = torch.einsum("i,j->ij", pos, inv_freq).to(dtype)
            self._cache[key] = (angles.cos(), angles.sin())
        return self._cache[key]

    def _apply_rope(self, tokens: Tensor, positions: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # positions: (B, T) long; tokens: (B, H, T, d); cos/sin: (max_pos, d/2)
        cos_e = torch.nn.functional.embedding(positions, cos)[:, None]
        sin_e = torch.nn.functional.embedding(positions, sin)[:, None]
        return rotate_pairs(tokens, cos_e, sin_e)

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

        # Table capacity from the SHAPE, never from the data. `positions` holds patch
        # coordinates of a gh x gw grid, so every coordinate is < max(gh, gw) <= gh*gw = T
        # -- the token count is always a valid upper bound, and reading it costs nothing.
        #
        # This used to be `int(positions.max().item()) + 1`, a blocking GPU->CPU
        # synchronisation on every single call. Under gradient checkpointing that sync
        # runs again during recomputation, which happens on the AUTOGRAD ENGINE'S BACKWARD
        # THREAD, and there it hung: a U4DR-G run stopped dead at step 230 with all 56
        # CPU threads asleep in futex_do_wait, the GPU reporting 100% utilisation and
        # allocated memory flat to the byte. The faulthandler stack put the backward
        # thread exactly on that `.item()`. It reproduced three times, at two different
        # memory footprints (9514 MiB and 8002 MiB), which is what ruled out the
        # allocator explanation it was first blamed on.
        #
        # Removing it is a strict win regardless of the hang: a device sync per RoPE call
        # serialises the whole pipeline, and these run once per block per layer.
        max_pos = positions.size(-2)
        # Each half (row-encoded, column-encoded) contributes d/4 rotation planes.
        cos, sin = self._freqs(
            tokens.size(-1) // 4, max_pos, tokens.device, tokens.dtype
        )

        y, x = tokens.chunk(2, dim=-1)
        y = self._apply_rope(y, positions[..., 0].long(), cos, sin)
        x = self._apply_rope(x, positions[..., 1].long(), cos, sin)
        return torch.cat((y, x), dim=-1)
