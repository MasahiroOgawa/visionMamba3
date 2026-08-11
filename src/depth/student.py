"""The depth student: a Vision Mamba-3 backbone feeding Depth-Anything-3's depth head.

Three pieces, and the middle one is the reason this file exists.

1. A DINOv2 ViT whose token mixer is one of ours (:mod:`visionmamba3`), built with
   ``cat_token=False`` so it produces a single ``embed_dim``-wide stream.
2. :class:`DimBridge`, a learnable ``embed_dim -> 2*embed_dim`` map, initialised so that
   at step 0 it is exactly ``cat([f, f], -1)``.
3. DA3's own DualDPT depth head, which expects ``2*embed_dim`` because DA3 concatenates a
   local and a global token stream.

Why the bridge. DA3's head was fitted to the output of softmax attention. Swapping the
mixer changes the basis those features live in, and without an adapter the fine-tune has
to absorb that mismatch by retraining the whole head on a few hundred images. The bridge
gives it one linear map to learn instead, starting from a no-op. Measured on ETH3D, that
difference is abs_rel 0.053 with the bridge against 0.159 without it, so it is not a
detail.

The static alternative -- literally ``cat([f, f], -1)`` -- feeds the head the same 384
numbers twice, halving the information it has to work with. The bridge starts there and
is free to move away.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn

_DA3_SRC = Path(__file__).resolve().parents[2] / "third_party" / "depth-anything-3" / "src"
if _DA3_SRC.exists() and str(_DA3_SRC) not in sys.path:
    sys.path.insert(0, str(_DA3_SRC))

from depth_anything_3.model.dinov2.layers.block import Block  # noqa: E402
from depth_anything_3.model.dinov2.vision_transformer import vit_small  # noqa: E402

from visionmamba3.self_attention import Mamba3SelfAttention  # noqa: E402
from visionmamba3.vssd_attention import (  # noqa: E402
    Mamba3VSSDAttention,
    Mamba3VSSDBetaGammaAttention,
)


class _MixerBlock(nn.Module):
    """Adapts one of our operators to the signature DINOv2's ``Block`` calls.

    DINOv2 constructs its attention as ``attn_class(dim, num_heads=..., qkv_bias=..., ...)``
    and calls it as ``attn(x, pos=...)``. Our operators take a different set of keyword
    arguments and ignore the softmax-specific ones, so the surplus is absorbed here rather
    than by loosening every operator's signature.
    """

    def __init__(self, dim: int, num_heads: int = 6, *, mixer: str = "vssd_bg",
                 state_dim: int = 64, chunk_size: Optional[int] = None,
                 rope: Optional[nn.Module] = None,
                 proj_bias: bool = True, use_fused_kernel: bool = False,
                 **_ignored) -> None:
        super().__init__()
        common = dict(dim=dim, num_heads=num_heads, state_dim=state_dim,
                      rope=rope, out_proj=True, proj_bias=proj_bias)
        if mixer == "bidirectional":
            # chunk_size matters only for the scan operators, which materialise mask rows.
            # At 504px T is 1296 and the full T x T mask is an order of magnitude slower.
            # The collapse variants have no T x T mask and ignore it.
            self.inner = Mamba3SelfAttention(bidirectional=True, three_term=True,
                                             chunk_size=chunk_size,
                                             use_fused_kernel=use_fused_kernel, **common)
        elif mixer == "vssd":
            self.inner = Mamba3VSSDAttention(**common)
        elif mixer == "vssd_bg":
            self.inner = Mamba3VSSDBetaGammaAttention(**common)
        else:
            raise ValueError(f"unknown mixer {mixer!r}; expected bidirectional|vssd|vssd_bg")

    def forward(self, x: Tensor, pos: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        return self.inner(x, pos=pos, attn_mask=attn_mask)


def flatten_views(feats: list[Tensor]) -> list[Tensor]:
    """(B, S, T, D) -> (B*S, T, D), for losses and metrics that score images not scenes."""
    return [f.flatten(0, 1) if f.dim() == 4 else f for f in feats]


class DimBridge(nn.Module):
    """Learnable ``in_dim -> 2*in_dim`` map, initialised to ``cat([x, x], -1)``.

    The identity init matters: it makes the untrained student numerically equal to the
    static-duplication fallback, so Phase-B distillation starts from a known point rather
    than from noise injected between the backbone and the head.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 2 * in_dim)
        with torch.no_grad():
            eye = torch.eye(in_dim)
            self.linear.weight.copy_(torch.cat([eye, eye], dim=0))
            self.linear.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class DepthStudent(nn.Module):
    """Vision Mamba-3 backbone + :class:`DimBridge` + DA3's DualDPT head.

    ``export_layers`` are the backbone depths whose features are handed to the head (and
    matched against the teacher during distillation). One bridge is kept per exported
    layer: they see differently-scaled features and sharing one map across them measurably
    under-fits.
    """

    def __init__(self, *, mixer: str = "vssd_bg", img_size: int = 504,
                 patch_size: int = 14, state_dim: int = 64, chunk_size: Optional[int] = 128,
                 export_layers: tuple[int, ...] = (5, 7, 9, 11),
                 use_fused_kernel: bool = False) -> None:
        super().__init__()
        self.mixer = mixer
        self.img_size = img_size
        self.export_layers = tuple(export_layers)

        # use_fused_kernel defaults to FALSE here, unlike the operator's own default.
        # The Triton kernel and the reference PyTorch path are not numerically equivalent:
        # on identical weights and input they differ by max 2.5 in absolute value, and the
        # reference implementation this pipeline was ported from has no fused path at all.
        # Loading that implementation's own checkpoint and evaluating it here reproduces its
        # reported 0.0531 only with the kernel off (0.0530); with it on the same weights give
        # 0.1160. Until the discrepancy is resolved, the reference path is what we train and
        # evaluate with. VSSD operators ignore this flag -- they never use the kernel.
        attn_class = partial(_MixerBlock, mixer=mixer, state_dim=state_dim,
                             chunk_size=chunk_size, use_fused_kernel=use_fused_kernel)
        self.vit = vit_small(
            img_size=img_size,
            patch_size=patch_size,
            block_fn=partial(Block, attn_class=attn_class),
            cat_token=False,   # one stream; the bridge widens it for the head
        )
        self.embed_dim = self.vit.embed_dim
        self.bridges = nn.ModuleList(DimBridge(self.embed_dim) for _ in self.export_layers)

    def init_from_da3(self, da3_model) -> dict:
        """Copy DA3's pretrained backbone weights in, skipping the attention it replaces.

        This is not optional. Only the mixer is trained in Phase B and only mixer+bridge in
        Phase C, so whatever is not copied here stays at its random initialisation for the
        whole run -- patch embedding, every MLP, every LayerNorm. A mixer can still drive the
        distillation loss down against a random patch embedding by compensating for it, which
        is why omitting this shows up as a generalisation gap on a held-out scene rather than
        as a training-loss failure.
        """
        src = da3_model.model.backbone.pretrained.state_dict()
        dst = self.vit.state_dict()
        keep, skipped_attn, mismatch, absent = {}, 0, 0, 0
        for k, v in dst.items():
            if ".attn." in k:
                skipped_attn += 1
            elif k not in src:
                absent += 1
            elif src[k].shape == v.shape:
                keep[k] = src[k]
            else:
                mismatch += 1
        self.vit.load_state_dict(keep, strict=False)
        return {"loaded": len(keep), "total": len(dst), "skipped_attn": skipped_attn,
                "shape_mismatch": mismatch, "missing_in_src": absent}

    @staticmethod
    def _as_multiview(images: Tensor) -> Tensor:
        """DA3's ViT is multi-view and wants (B, S, 3, H, W). Accept (B, 3, H, W) too.

        Made explicit rather than reshaping silently: a single-view call is a legitimate
        use (the depth head scores one image at a time) but a caller who passes a batch of
        views as a batch of images would otherwise get plausible-looking wrong answers.
        """
        if images.dim() == 5:
            return images
        if images.dim() == 4:
            # (N, 3, H, W) is N views of ONE scene -> (1, N, 3, H, W), not N scenes of one
            # view. The distinction is not cosmetic: cross-view attention and
            # ref_view_strategy="first" both operate along S, so folding views into B
            # would silently give each image its own reference and remove the cross-view
            # signal. The teacher is unsqueezed the same way in depth/da3.py.
            return images.unsqueeze(0)
        raise ValueError(f"expected (B, S, 3, H, W) or (N, 3, H, W); got {tuple(images.shape)}")

    def features(self, images: Tensor) -> list[Tensor]:
        """Per-exported-layer patch tokens, before the bridge. (B, S, T, embed_dim) each.

        Taken from the ``aux`` output via ``export_feat_layers``, which is the
        single-stream ``embed_dim`` path -- the same call the teacher uses, so distillation
        compares like with like. The main ``n=`` output would instead give ``2*embed_dim``
        once ``cat_token`` is on, which the teacher has and we do not.

        The view axis is kept: DualDPT unpacks ``B, S, N, C`` and fails if views have been
        folded into the batch. Use :func:`flatten_views` where a per-image view is wanted.
        """
        _main, aux = self.vit.get_intermediate_layers(
            self._as_multiview(images), n=1,
            export_feat_layers=list(self.export_layers), ref_view_strategy="first",
        )
        return list(aux)

    def bridged(self, images: Tensor) -> list[Tensor]:
        """Per-exported-layer tokens after the bridge. (B, S, T, 2*embed_dim) each."""
        return [b(f) for b, f in zip(self.bridges, self.features(images))]

    def trainable(self, scope: str) -> list[nn.Parameter]:
        """Parameters for one training scope, so callers do not re-derive the split.

        ``mixer`` is Phase-B (the operator alone); ``bridge`` and ``all`` are Phase-C.
        """
        if scope == "mixer":
            return [p for n, p in self.named_parameters() if ".inner." in n]
        if scope == "bridge":
            return list(self.bridges.parameters())
        if scope == "all":
            return list(self.parameters())
        raise ValueError(f"unknown scope {scope!r}; expected mixer|bridge|all")
