"""ViT-Tiny CIFAR-10 skeleton with a pluggable token mixer.

Ported from the (now-retired) ``largescale3Dreconstruction_using_SSM`` harness
``scripts/cifar10_compare.py`` so the operator comparison in Table 1 of the
vmamba3 paper can be reproduced and extended (the NC-SSD long-T row). The
skeleton (patch embed, CLS token, learnable pos-embed, pre-norm blocks, head)
is kept byte-faithful to that harness so the already-published rows reproduce;
only the token mixer is swapped:

  vit_attn         -> VanillaAttention (softmax)
  vit_mamba3       -> Mamba3SelfAttention, 2-directional (row-major fwd+rev)
  vit_mamba3_4dir  -> Mamba3SelfAttention, 4-directional (+ column-major)
  vit_mamba3_vssd  -> Mamba3VSSDAttention (VSSD-gamma, non-causal)
  vit_mamba3_vssd_bg -> Mamba3VSSDBetaGammaAttention (VSSD-beta,gamma; two pools)

The mamba mixers are built exactly as the retired ``da3_adapter`` wrappers
built them, and are placed as ``block.attn`` in place of the softmax module,
mirroring the original ``install_mamba3`` swap (same RNG order and kwargs).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Importing the subpackage triggers visionmamba3/__init__ (mamba-ssm path
# injection needed by the fused SSD kernel path).
from visionmamba3.rope2d import RoPE2D
from visionmamba3.self_attention import Mamba3SelfAttention
from visionmamba3.vssd_attention import (
    Mamba3VSSDAttention,
    Mamba3VSSDBetaGammaAttention,
)

# The direction count lives in the variant name rather than in a flag: these are
# separate Table 1 rows, so a run must be able to produce both in one sweep, and
# the name is then the single source of truth for what a row measured.
VARIANTS = ("cnn", "vit_attn", "vit_mamba3", "vit_mamba3_4dir", "vit_mamba3_vssd",
            "vit_mamba3_vssd_bg")
_MAMBA3_DIRECTIONS = {"vit_mamba3": 2, "vit_mamba3_4dir": 4}

# Whether Mamba-3's internal complex rotary is on by default, per operator.
# Measured at T=65 with the rotary's and 2-D RoPE's rotation planes matched:
#   2-dir    82.71 -> 81.89  (-0.82)
#   4-dir    83.26 -> 81.47  (-1.79)
#   NC-SSD   79.40 -> 81.18  (+1.78)
# The scan operators already carry relative position in the SSD decay mask's
# |i-j| band, so a second relative encoding is redundant there and only
# interferes. NC-SSD's mask collapses to a per-token scalar and has no |i-j|
# term at all, so the rotary is new information for it -- the same reason it
# gains most from 2-D RoPE. Override per run with --rope-angles/--no-rope-angles.
_MAMBA3_ROTARY_DEFAULT = {
    "vit_mamba3": False,
    "vit_mamba3_4dir": False,
    "vit_mamba3_vssd": True,
    # VSSD-beta,gamma inherits VSSD-gamma's reason for wanting the rotary: both
    # pools' masks are per-token scalars with no |i-j| term, so the rotary is
    # still their only relative-position signal. Untested for this variant --
    # the 3x3 grid measures it.
    "vit_mamba3_vssd_bg": True,
}


# Both non-causal collapses take the same constructor kwargs and differ only in
# how many global pools they read, so one build branch serves both.
_VSSD_CLASSES = {
    "vit_mamba3_vssd": (Mamba3VSSDAttention, "VSSD-gamma"),
    "vit_mamba3_vssd_bg": (Mamba3VSSDBetaGammaAttention, "VSSD-beta,gamma"),
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class SmallResNet(nn.Module):
    """CIFAR-style ResNet: 3x3 stem + 3 stages x 2 BasicBlocks at {64, 128, 256}."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.stage2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.stage3 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.gap(x).flatten(1)


class VanillaAttention(nn.Module):
    """Timm-style multi-head softmax attention (the swap target for the mixers)."""

    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        del pos  # softmax attention takes position from the additive pos-embed
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    """Pre-norm transformer block. ``self.attn`` is the mixer swap target."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = VanillaAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        # Set by _swap_mixer when the installed mixer runs column-major scans.
        self.attn_takes_grid = False

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None,
                grid: tuple[int, int] | None = None) -> torch.Tensor:
        kw = {"grid": grid} if self.attn_takes_grid else {}
        x = x + self.attn(self.norm1(x), pos, **kw)
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTiny(nn.Module):
    """ViT-Tiny@d6: patch embed (32->grid tokens), CLS, learnable pos-emb."""

    def __init__(self, img_size: int = 32, patch: int = 4, dim: int = 192,
                 depth: int = 6, num_heads: int = 3, mlp_ratio: float = 4.0):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        n_patches = (img_size // patch) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.blocks = nn.ModuleList([Block(dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.feat_dim = dim
        self.num_heads = num_heads
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # (y, x) grid coordinates per token, for mixers that take 2-D RoPE. Patch
        # coordinates start at 1 so the CLS token, which has no place on the grid,
        # can hold (0, 0) without colliding with the top-left patch.
        g = img_size // patch
        self.grid = (g, g)
        rc = torch.arange(g)
        coords = torch.stack(torch.meshgrid(rc, rc, indexing="ij"), dim=-1).view(-1, 2) + 1
        token_pos = torch.cat([torch.zeros(1, 2, dtype=coords.dtype), coords])
        self.register_buffer("token_pos", token_pos.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        pos = self.token_pos.expand(B, -1, -1)
        for blk in self.blocks:
            x = blk(x, pos, grid=self.grid)
        x = self.norm(x)
        return x[:, 0]


class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 10):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def _swap_mixer(vit: ViTTiny, make_mixer) -> int:
    """Replace every block's softmax ``attn`` with a freshly built mixer.

    Mirrors the retired ``install_mamba3(which="backbone_only")`` swap: ViTTiny
    is constructed with VanillaAttention first (same RNG draw as the softmax
    variant), then each ``block.attn`` is overwritten in place.
    """
    n = 0
    dim, num_heads = vit.pos_embed.shape[-1], vit.num_heads
    for blk in vit.blocks:
        blk.attn = make_mixer(dim, num_heads)
        n += 1
    return n


def build_model(
    variant: str, patch_size: int = 4, num_classes: int = 10, rope: bool = True,
    fused: bool = True, rope_angles: bool | None = None,
    rope_angle_scale: float = 1.0,
    rope_turns: float | None = None,
) -> nn.Module:
    """Build a CIFAR-10 classifier for one operator variant.

    The mamba mixers use the same kwargs the retired da3_adapter wrappers used:
    ``state_dim=64, out_proj=True, proj_bias=True``; bidirectional SSD keeps
    ``bidirectional=True, three_term=True``; NC-SSD is non-causal (no mask).

    ``rope`` gives the mamba mixers 2-D RoPE on their B/C projections. It matters
    far more to NC-SSD than to bidirectional SSD: bidirectional keeps the
    |i-j| decay mask, which is already a relative-position signal, whereas the
    NC-SSD collapse leaves RoPE as its only source of relative position. The
    originally published Table 1 rows were run without it (``--no-rope``
    reproduces them); it is on by default because that is the operator the paper
    describes. RoPE2D holds no parameters, so enabling it does not disturb the
    init RNG order that keeps the variants comparable.

    ``rope_angles`` turns on Mamba-3's own complex-SSM rotary (a learned relative
    encoding inside the operator), which stacks with ``rope``, the external
    absolute 2-D encoding. ``None`` takes the per-operator default from
    ``_MAMBA3_ROTARY_DEFAULT``: it pays for NC-SSD and costs the scan operators.

    The scan-direction count comes from the variant name (see ``VARIANTS``).
    NC-SSD has none: its mask collapses, so there is no scan order.

    ``fused`` selects Mamba3SelfAttention's Triton kernel (NC-SSD never uses it),
    which is both the faster and the reference path's numerical better; keep it on.
    ``--no-fused`` falls back to the equivalent PyTorch path for CPU runs or
    kernel debugging. Either way we pass ``row_renorm=False``: the kernel silently
    does not implement row renormalisation, so the constructor default of True
    would make the two paths compute *different* operators -- a trap worth
    keeping closed, since published numbers came from the kernel.
    """
    if variant == "cnn":
        return Classifier(SmallResNet(), num_classes)
    if variant == "vit_attn":
        return Classifier(ViTTiny(patch=patch_size), num_classes)
    # One shared instance: RoPE2D is parameter-free and caches its angle tables
    # per (dim, max_pos), which every block here hits identically.
    rope_mod = RoPE2D(base_frequency=100.0) if rope else None
    tag = "RoPE2D" if rope else "no RoPE"
    if rope_angles is None:
        rope_angles = _MAMBA3_ROTARY_DEFAULT.get(variant, False)
    if variant in _MAMBA3_DIRECTIONS:
        num_directions = _MAMBA3_DIRECTIONS[variant]
        model = Classifier(ViTTiny(patch=patch_size), num_classes)
        n = _swap_mixer(
            model.backbone,
            lambda dim, h: Mamba3SelfAttention(
                dim, num_heads=h, state_dim=64, bidirectional=True,
                three_term=True, out_proj=True, proj_bias=True, rope=rope_mod,
                use_fused_kernel=fused, row_renorm=False,
                rope_angles=rope_angles, num_directions=num_directions,
            ),
        )
        if num_directions == 4:
            for blk in model.backbone.blocks:
                blk.attn_takes_grid = True
        kern = "Triton kernel" if fused else "reference path"
        ang = "+rotary" if rope_angles else "no rotary"
        print(f"  [{variant}] swapped {n} attention modules -> Mamba3SelfAttention "
              f"({num_directions}-dir SSD, {tag}, {ang}, {kern})")
        return model
    if variant in _VSSD_CLASSES:
        cls, label = _VSSD_CLASSES[variant]
        model = Classifier(ViTTiny(patch=patch_size), num_classes)
        n = _swap_mixer(
            model.backbone,
            lambda dim, h: cls(
                dim, num_heads=h, state_dim=64, out_proj=True, proj_bias=True,
                rope=rope_mod, rope_angles=rope_angles,
                rope_angle_scale=rope_angle_scale, rope_turns=rope_turns,
            ),
        )
        ang = "+rotary" if rope_angles else "no rotary"
        print(f"  [{variant}] swapped {n} attention modules -> "
              f"{cls.__name__} ({label}, {tag}, {ang})")
        return model
    raise ValueError(f"unknown variant: {variant}")
