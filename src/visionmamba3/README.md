# `mamba3_attn.mamba3` — Mamba-3 SSD attention library

The core research contribution of this project: a Mamba-3 Structured
State-space Duality (SSD) attention module that can be **dropped into
any transformer architecture** as a replacement for standard self /
cross attention.

## Design

Standard transformer self-attention computes `Y = softmax(QKᵀ) V` with
O(T²) memory and time. Mamba-3 SSD attention computes

```
Y = (L ⊙ (C Bᵀ)) V
```

where:
- `B, C` are query / key-like projections (`(B, T, num_heads, state_dim)`).
- `L` is a **structured** mask whose entries are products of decay
  scalars `λ_t`, not learned attention scores. Two- and three-term
  variants are provided (`build_two_term_mask`, `build_three_term_mask`).
- `V` is the value projection (`(B, T, num_heads, head_dim)`).

The structured `L` admits O(T·N) chunked scan implementations (provided
by the upstream `mamba_ssm` Triton kernel), giving linear-in-T cost vs
transformer's quadratic.

The library implements:
- **SISO** — single-input single-output self-attention
  (`Mamba3SelfAttention`). Bidirectional mode runs forward + backward
  scans and sums them.
- **MIMO** — multi-input multi-output cross-attention
  (`Mamba3CrossAttention`). For encoder-decoder patterns.
- **Three-term decay** — adds a third structured term that empirically
  helps stabilize the attention pattern at long T.
- **2D RoPE** — `RoPE2D` matches the rotary embedding convention used
  by DA3's attention (so a swap preserves position-encoding behavior).

## Usage — standalone

```python
from mamba3_attn.mamba3 import Mamba3SelfAttention

# Drop-in replacement: same (B, T, C) → (B, T, C) signature.
attn = Mamba3SelfAttention(
    dim=384, num_heads=6, state_dim=64,
    bidirectional=True, three_term=True,
)
y = attn(x)  # x: (B, T, 384)
```

## Usage — Depth-Anything-3 swap

The integration is in `mamba3_attn.patch` and `mamba3_attn.da3_adapter`:

```python
from depth_anything_3.api import DepthAnything3
from mamba3_attn.patch import install_mamba3

api = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
n_swapped = install_mamba3(api.model, which="all", state_dim=64)
# n_swapped == 16 for DA3-SMALL (12 backbone blocks + 4 cam_enc blocks)

# api.model now has Mamba-3 attention everywhere a transformer
# Attention used to live; everything else (DPT head, cam_dec, cam_enc
# projection layers) is untouched.
```

## Usage — generic ViT

For any model with `block.attn` modules following the
`(dim, num_heads, qkv_bias=..., proj_bias=..., rope=..., attn_drop=...,
proj_drop=...)` constructor signature and `(x, pos=..., attn_mask=...)`
forward signature (DA3 / DINOv2 / timm convention):

```python
from mamba3_attn.da3_adapter import Mamba3Attention
for block in model.backbone.blocks:
    block.attn = Mamba3Attention(
        dim=block.attn.qkv.in_features,
        num_heads=block.attn.num_heads,
        rope=getattr(block.attn, "rope", None),
        state_dim=64,
    )
```

## Optional: fused Triton kernel

When `use_fused_kernel=True`, `Mamba3SelfAttention` routes through
`mamba_ssm.ops.triton.mamba3.mamba3_siso_combined` for ~2.7× speedup at
inference. Verified numerically equivalent to the PyTorch chunked-scan
path within fp16 tolerance.

## Files

| file | purpose |
|---|---|
| `self_attention.py` | `Mamba3SelfAttention` — SISO bidirectional |
| `cross_attention.py` | `Mamba3CrossAttention` — MIMO cross |
| `projections.py` | `AttentionProjections`, `BCNorm` |
| `mask.py` | structured decay mask builders |
| `rope2d.py` | 2D rotary position embedding |
