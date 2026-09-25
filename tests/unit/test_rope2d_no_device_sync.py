"""RoPE2D must not synchronise the device.

`max_pos` used to come from `int(positions.max().item()) + 1`. Under gradient
checkpointing that blocking GPU->CPU sync re-runs during recomputation, on the autograd
engine's backward thread, where it deadlocked: a 1011M-parameter run stopped at step 230
with every CPU thread asleep in futex_do_wait and the GPU pinned at 100%.

`torch.cuda.set_sync_debug_mode("error")` turns any synchronisation into an exception, so
this fails loudly if the sync ever comes back.
"""

import pytest
import torch

from visionmamba3.rope2d import RoPE2D


def _grid(gh: int, gw: int, device) -> torch.Tensor:
    ys, xs = torch.arange(gh, device=device), torch.arange(gw, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)[None]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA to detect a sync")
def test_forward_does_not_synchronise_the_device():
    rope = RoPE2D().cuda()
    pos = _grid(14, 46, "cuda")
    tokens = torch.randn(1, 2, 14 * 46, 64, device="cuda")
    rope(tokens, pos)                       # build the table outside the guard
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        rope(tokens, pos)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("gh,gw", [(12, 40), (14, 46), (16, 52)])
def test_table_capacity_from_shape_covers_every_coordinate(gh, gw):
    """T = gh*gw >= max(gh, gw), so the shape-derived bound is always sufficient."""
    pos = _grid(gh, gw, "cpu")
    assert int(pos.max()) < pos.size(-2)
    out = RoPE2D()(torch.randn(1, 2, gh * gw, 64), pos)
    assert out.shape == (1, 2, gh * gw, 64) and torch.isfinite(out).all()
