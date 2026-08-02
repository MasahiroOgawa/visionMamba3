"""Mamba-3's complex-SSM rotary (the kernel's `Angles` path).

The rotary is easy to get subtly wrong in ways that do not raise: the pairing
convention can disagree with the kernel, and a non-identity init can scramble
the similarity structure. Both are pinned here.
"""

from __future__ import annotations

import math

import pytest
import torch

from visionmamba3.projections import AttentionProjections
from visionmamba3.self_attention import Mamba3SelfAttention, apply_cumulative_rope


def test_rope_pairs_are_interleaved_not_half_split():
    """The kernel does `tl.split(tl.reshape(k, [T, N//2, 2]))` -- adjacent
    channels form a pair. A half-split convention also "works" numerically but
    silently computes a different operator, so pin the layout explicitly."""
    t = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])  # (1,1,1,4): pairs (1,0), (1,0)
    theta = torch.full((1, 1, 1, 2), math.pi / 2)  # quarter turn on both pairs
    out = apply_cumulative_rope(t, theta).squeeze()
    # (1,0) rotated by +90 deg -> (0,1), written back interleaved.
    assert torch.allclose(out, torch.tensor([0.0, 1.0, 0.0, 1.0]), atol=1e-6)


def test_zero_angles_are_the_identity():
    t = torch.randn(2, 3, 5, 8)
    out = apply_cumulative_rope(t, torch.zeros(2, 3, 5, 4))
    assert torch.allclose(out, t, atol=1e-6)


def test_angle_head_is_zero_initialised_but_learnable():
    """Zero-init keeps the rotary at identity on step 0 (so it is a safe add-on
    to pretrained weights) without killing its gradient -- the derivative of a
    rotation at theta=0 is a quarter turn, not zero."""
    proj = AttentionProjections(64, num_heads=4, state_dim=16, rope_angles=True)
    x = torch.randn(2, 7, 64)
    *_, angles = proj(x)
    assert angles is not None
    assert torch.count_nonzero(angles) == 0, "rotary must start as identity"

    # The gradient has to be taken through the *rotation*, not through the angle
    # values: d(angle^2)/dW = 2 * angle * x is zero when the head is zero-init,
    # which would look like a dead head while the rotary is in fact learnable.
    attn = Mamba3SelfAttention(
        dim=64, num_heads=4, state_dim=16, rope_angles=True, use_fused_kernel=False
    )
    attn(torch.randn(2, 7, 64)).square().mean().backward()
    grad = attn.projections.proj.weight.grad[-attn.projections.num_rope_angles :]
    assert grad.abs().max() > 0, "angle head must still receive gradient"


def test_rope_angles_off_keeps_the_projection_width():
    """Existing checkpoints must keep loading, so the default must not widen the
    fused projection."""
    off = AttentionProjections(64, num_heads=4, state_dim=16)
    on = AttentionProjections(64, num_heads=4, state_dim=16, rope_angles=True)
    assert off.num_rope_angles == 0
    assert on.out_size - off.out_size == 16 // 2
    assert off(torch.randn(1, 3, 64))[-1] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel path needs CUDA")
def test_reference_matches_kernel_with_rope_angles():
    pytest.importorskip(
        "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined",
        reason="fused kernel unavailable (is the mamba-ssm submodule checked out?)",
    )
    torch.manual_seed(0)
    common = dict(
        dim=384, num_heads=6, state_dim=64, bidirectional=False, three_term=True,
        rope=None, post_norm=False, out_proj=False, row_renorm=False, rope_angles=True,
    )
    ref = Mamba3SelfAttention(**common, use_fused_kernel=False).cuda().eval()
    ker = Mamba3SelfAttention(**common, use_fused_kernel=True).cuda().eval()
    ker.load_state_dict(ref.state_dict())
    x = torch.randn(2, 128, 384, device="cuda")
    with torch.no_grad():
        cos = torch.nn.functional.cosine_similarity(
            ref(x).flatten(), ker(x).flatten(), dim=0
        ).item()
    assert cos >= 0.95, f"reference and kernel rotary disagree (cos={cos:.4f})"
