"""Loading Depth-Anything-3 and driving its depth head.

Two things here that are not obvious from DA3's own code.

``_stub_export_module`` — DA3's ``api.py`` does
``from depth_anything_3.utils.export import export``, which transitively imports pycolmap,
moviepy, gsplat and trimesh. None is needed for depth inference or feature extraction, and
the submodule is upstream's and must not be edited, so a no-op stub is registered in
``sys.modules`` before the import runs.

``dualdpt_depth`` — the head is called as ``head(feats, H, W, patch_start_idx=0)`` where
``feats`` is a list of *one-tuples*, because DualDPT indexes ``feats[i][0]``. It returns a
dict whose depth key is named by ``head.head_main``. Both are easy to get wrong in a way
that raises deep inside the head rather than at the call site.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

_DA3_SRC = Path(__file__).resolve().parents[2] / "third_party" / "depth-anything-3" / "src"
if _DA3_SRC.exists() and str(_DA3_SRC) not in sys.path:
    sys.path.insert(0, str(_DA3_SRC))

DEFAULT_DA3 = "depth-anything/DA3-SMALL"


def _stub_export_module() -> None:
    name = "depth_anything_3.utils.export"
    if name in sys.modules:
        return
    stub = types.ModuleType(name)
    stub.export = lambda *a, **kw: None
    sys.modules[name] = stub


def load_da3(hf_model: str = DEFAULT_DA3, device: str = "cpu"):
    """Load a pretrained DA3. Used both as the distillation teacher and for its head."""
    _stub_export_module()
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(hf_model).to(device)
    model.eval()
    model.device = device
    return model


def dualdpt(da3_model) -> nn.Module:
    """DA3's DualDPT depth head."""
    return da3_model.model.head


def dualdpt_depth(head: nn.Module, feats: Sequence[Tensor], height: int, width: int) -> Tensor:
    """Run bridged features through DualDPT. Returns (N, H, W) depth in metres.

    ``feats`` must be exactly four tensors of (N, T, 2*embed_dim) -- DualDPT's
    intermediate-layer contract -- and is wrapped as one-tuples because the head indexes
    ``feats[i][0]``.
    """
    if len(feats) != 4:
        raise ValueError(f"DualDPT expects exactly 4 feature layers, got {len(feats)}")
    out = head([(f,) for f in feats], height, width, patch_start_idx=0)
    depth = out[getattr(head, "head_main", "depth")]
    # (B, S, H, W) or (B, S, 1, H, W) -> (B*S, H, W): metrics score images, not scenes.
    if depth.dim() == 5:
        depth = depth.squeeze(2)
    return depth.flatten(0, 1) if depth.dim() == 4 else depth


@torch.no_grad()
def teacher_features(da3_model, images: Tensor, layers: Sequence[int]) -> list[Tensor]:
    """Teacher patch tokens at ``layers``, shaped (B, S, T, embed_dim).

    Uses the ``aux`` / ``export_feat_layers`` path, which is single-stream ``embed_dim``
    (384 for DA3-SMALL) and therefore the same width as the student. The main ``n=``
    output is ``2*embed_dim`` because the teacher ships with ``cat_token`` on, and matching
    against that would compare a 384-d student to a 768-d teacher.
    """
    vit = da3_model.model.backbone.pretrained
    x = images if images.dim() == 5 else images.unsqueeze(0)
    _main, aux = vit.get_intermediate_layers(
        x, n=1, export_feat_layers=list(layers), ref_view_strategy="first",
    )
    return list(aux)
