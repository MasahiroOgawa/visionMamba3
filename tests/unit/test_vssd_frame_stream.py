"""Frame-causal streaming form of VSSD-2pool (Mamba3VSSD2PoolAttention.forward_frames).

What must hold for a window-free stream to be trustworthy:
* one call on n frames equals n calls on one frame with the state carried (so a stream
  of any length is the same computation as a clip);
* with decay 1, the LAST frame of an n-frame clip equals the windowed operator on that
  clip, frame offset ``y + v*gh`` included (so a windowed checkpoint warm-starts it);
* the state does not grow with the number of frames.
"""

from __future__ import annotations

import torch

from visionmamba3.rope2d import RoPE2D
from visionmamba3.vssd_attention import Mamba3VSSD2PoolAttention

GH, GW = 3, 4                       # token grid per frame
T = GH * GW


def _mk(ref: int, gamma_one: bool = False):
    torch.manual_seed(0)
    attn = Mamba3VSSD2PoolAttention(dim=32, num_heads=4, state_dim=8, rope=RoPE2D()).double()
    torch.nn.init.normal_(attn.proj2.weight, std=0.2)      # both pools active
    attn.enable_frame_stream(ref)
    attn.double()
    if gamma_one:
        with torch.no_grad():
            attn.frame_decay.fill_(-1e4)                    # softplus -> 0, gamma -> 1
    return attn


def _grid(n: int, offset: bool) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(GH), torch.arange(GW), indexing="ij")
    g = torch.stack([ys.flatten(), xs.flatten()], -1)
    return torch.cat([g + torch.tensor([v * GH if offset else 0, 0]) for v in range(n)])[None]


def test_one_call_equals_frame_by_frame():
    attn = _mk(ref=2)
    n = 5
    x = torch.randn(1, n * T, 32, dtype=torch.float64)
    y_all, st_all = attn.forward_frames(x, _grid(n, False), n, GH)
    state, ys = None, []
    for v in range(n):
        y, state = attn.forward_frames(x[:, v * T:(v + 1) * T], _grid(1, False), 1, GH, state)
        ys.append(y)
    torch.testing.assert_close(torch.cat(ys, 1), y_all, rtol=1e-12, atol=1e-12)
    for a, b in zip(state, st_all):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_last_frame_equals_windowed_operator():
    for n in (2, 4):
        attn = _mk(ref=n, gamma_one=True)
        x = torch.randn(2, n * T, 32, dtype=torch.float64)
        windowed = attn(x, pos=_grid(n, True).expand(2, -1, -1))
        streamed, _ = attn.forward_frames(x, _grid(n, False).expand(2, -1, -1), n, GH)
        # RoPE2D builds its angle table in float32, the stream rotation in float64, so they
        # agree to float32 precision, not to float64.
        torch.testing.assert_close(streamed[:, -T:], windowed[:, -T:], rtol=1e-5, atol=1e-7)


def test_earlier_frames_do_not_read_the_future():
    attn = _mk(ref=2)
    x = torch.randn(1, 3 * T, 32, dtype=torch.float64)
    y, _ = attn.forward_frames(x, _grid(3, False), 3, GH)
    x2 = x.clone()
    x2[:, 2 * T:] += 1.0
    y2, _ = attn.forward_frames(x2, _grid(3, False), 3, GH)
    torch.testing.assert_close(y[:, :2 * T], y2[:, :2 * T], rtol=0, atol=0)
    assert not torch.allclose(y[:, 2 * T:], y2[:, 2 * T:])


def test_state_size_is_independent_of_stream_length():
    attn = _mk(ref=2)
    sizes = []
    for n in (1, 50):
        _, st = attn.forward_frames(torch.randn(1, n * T, 32, dtype=torch.float64),
                                    _grid(n, False), n, GH)
        sizes.append(sum(t.numel() for t in st))
    assert sizes[0] == sizes[1]


def test_read_scale_stays_bounded_on_a_long_stream():
    attn = _mk(ref=2, gamma_one=True)
    x = torch.randn(1, T, 32, dtype=torch.float64)
    state, norms = None, []
    for _ in range(200):                                    # the same frame, repeated
        y, state = attn.forward_frames(x, _grid(1, False), 1, GH, state)
        norms.append(float(y.detach().norm()))
    assert max(norms) < 10 * norms[1]


def test_decay_receives_gradient():
    attn = _mk(ref=2).float()
    y, _ = attn.forward_frames(torch.randn(1, 3 * T, 32), _grid(3, False), 3, GH)
    y.square().sum().backward()
    assert attn.frame_decay.grad is not None and attn.frame_decay.grad.abs().sum() > 0
