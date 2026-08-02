"""4-directional SSD: row-major forward/backward plus column-major both ways.

A raster scan puts vertical neighbours T apart in scan order while horizontal
ones are adjacent, so the decay mask sees a strongly anisotropic image. The
column-major pair restores the other axis.
"""

from __future__ import annotations

import pytest
import torch

from visionmamba3.self_attention import Mamba3SelfAttention

COMMON = dict(
    dim=32, num_heads=4, state_dim=8, three_term=True,
    post_norm=False, out_proj=False, row_renorm=False, use_fused_kernel=False,
)


def test_column_major_permutation_round_trips_and_pins_prefix():
    idx, inv = Mamba3SelfAttention._column_major_index((2, 3), T=7, device="cpu")
    # 2x3 row-major body at indices 1..6 -> column-major visits rows within a
    # column first: (r0c0, r1c0, r0c1, r1c1, r0c2, r1c2).
    assert idx.tolist() == [0, 1, 4, 2, 5, 3, 6]
    assert torch.equal(idx[inv], torch.arange(7)), "inverse must undo the permutation"
    assert idx[0].item() == 0, "the CLS token has no (row, col) and must stay put"


def test_grid_must_fit_the_token_count():
    with pytest.raises(ValueError, match="needs 12 tokens"):
        Mamba3SelfAttention._column_major_index((3, 4), T=5, device="cpu")


def test_four_directional_equals_bidirectional_at_init():
    """col_gates are zero-init, so the extra directions contribute nothing until
    trained -- switching to 4 directions is a strict add-on, never a regression."""
    torch.manual_seed(0)
    bi = Mamba3SelfAttention(**COMMON, bidirectional=True, num_directions=2).eval()
    torch.manual_seed(0)
    quad = Mamba3SelfAttention(**COMMON, bidirectional=True, num_directions=4).eval()
    quad.load_state_dict(bi.state_dict(), strict=False)

    x = torch.randn(2, 1 + 4 * 4, 32)
    with torch.no_grad():
        assert torch.allclose(bi(x), quad(x, grid=(4, 4)), atol=1e-6)


def test_four_directional_needs_a_grid():
    quad = Mamba3SelfAttention(**COMMON, bidirectional=True, num_directions=4)
    with pytest.raises(ValueError, match="needs the token grid"):
        quad(torch.randn(1, 17, 32))


def test_column_gates_receive_gradient():
    quad = Mamba3SelfAttention(**COMMON, bidirectional=True, num_directions=4)
    quad(torch.randn(2, 1 + 4 * 4, 32), grid=(4, 4)).square().mean().backward()
    assert quad.col_gates.grad is not None
    assert quad.col_gates.grad.abs().max() > 0


def test_four_directions_actually_change_the_output_once_gated_open():
    torch.manual_seed(0)
    quad = Mamba3SelfAttention(**COMMON, bidirectional=True, num_directions=4).eval()
    x = torch.randn(2, 1 + 4 * 4, 32)
    with torch.no_grad():
        closed = quad(x, grid=(4, 4))
        quad.col_gates.fill_(1.0)
        opened = quad(x, grid=(4, 4))
    assert not torch.allclose(closed, opened, atol=1e-5)


def test_num_directions_is_validated():
    with pytest.raises(ValueError, match="must be 1, 2 or 4"):
        Mamba3SelfAttention(**COMMON, num_directions=3)
    with pytest.raises(ValueError, match="implies bidirectional"):
        Mamba3SelfAttention(**COMMON, bidirectional=False, num_directions=4)
