"""Unit tests for mamba3_attn.mamba3.rope2d.RoPE2D"""

from __future__ import annotations

import torch

from visionmamba3.rope2d import RoPE2D


def _positions(height: int, width: int) -> torch.Tensor:
    ys = torch.arange(height)
    xs = torch.arange(width)
    return torch.cartesian_prod(ys, xs).view(1, height * width, 2)


def test_rope_preserves_shape():
    rope = RoPE2D()
    tokens = torch.randn(2, 4, 16, 32)  # B, H, T, d ; d=32, d/2=16, d/4=8 ok
    pos = _positions(4, 4).expand(2, -1, -1)
    out = rope(tokens, pos)
    assert out.shape == tokens.shape


def test_rope_preserves_norm_per_token():
    rope = RoPE2D()
    tokens = torch.randn(1, 1, 9, 32)
    pos = _positions(3, 3)
    out = rope(tokens, pos)
    # RoPE is a rotation → per-token L2 norm preserved.
    n_in = tokens.pow(2).sum(dim=-1)
    n_out = out.pow(2).sum(dim=-1)
    assert torch.allclose(n_in, n_out, atol=1e-4)


def test_rope_identity_at_origin():
    rope = RoPE2D()
    tokens = torch.randn(1, 2, 1, 16)
    pos = torch.zeros(1, 1, 2, dtype=torch.long)
    out = rope(tokens, pos)
    # At position (0,0), all angles are 0 → cos=1, sin=0 → identity rotation.
    assert torch.allclose(out, tokens, atol=1e-6)


def test_rope_consistent_same_position():
    rope = RoPE2D()
    t = torch.randn(1, 1, 2, 16)
    pos = torch.tensor([[[3, 2], [3, 2]]], dtype=torch.long)  # both tokens at same coord
    # Both tokens get the same rotation → same-position tokens remain equal if they were equal.
    t_same = torch.stack([t[0, 0, 0], t[0, 0, 0]], dim=0)  # two copies of token 0
    t_same = t_same.view(1, 1, 2, 16)
    out_same = rope(t_same, pos)
    assert torch.allclose(out_same[0, 0, 0], out_same[0, 0, 1], atol=1e-5)
