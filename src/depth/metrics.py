"""Depth metrics, median-aligned.

Monocular depth is compared after median alignment because the prediction and the ground
truth need not share a scale: a model can have the shape exactly right and still be off by
a global factor. Aligning by the median ratio over valid pixels is the MiDaS/DA3
convention, and it is applied to *both* sides of any comparison so no method gets an
advantage from it.

``abs_rel`` is the headline. It is a per-pixel *relative* error, so a metre of error at
30 m counts far less than a metre at 1 m -- which is what makes it usable across scenes at
very different depths.
"""

from __future__ import annotations

import torch
from torch import Tensor


def median_align(pred: Tensor, gt: Tensor, valid: Tensor) -> Tensor:
    """Scale ``pred`` so its median matches ``gt``'s over valid pixels, per image."""
    out = pred.clone()
    for i in range(pred.shape[0]):
        v = valid[i]
        if v.sum() < 16:            # too few points for a stable median
            continue
        ratio = torch.median(gt[i][v]) / torch.median(pred[i][v].clamp_min(1e-6))
        out[i] = pred[i] * ratio
    return out


def depth_metrics(pred: Tensor, gt: Tensor, valid: Tensor, *, align: bool = True) -> dict:
    """Per-image metrics averaged over images. All inputs (N, H, W); ``valid`` is bool.

    Averaging per image rather than pooling every pixel keeps one large, densely-annotated
    image from dominating the score.
    """
    if align:
        pred = median_align(pred, gt, valid)
    acc: dict[str, list[float]] = {k: [] for k in ("abs_rel", "rmse", "log10", "delta_1_25")}
    for i in range(pred.shape[0]):
        v = valid[i]
        if v.sum() < 16:
            continue
        p = pred[i][v].clamp_min(1e-6).double()
        g = gt[i][v].clamp_min(1e-6).double()
        acc["abs_rel"].append(((p - g).abs() / g).mean().item())
        acc["rmse"].append(((p - g) ** 2).mean().sqrt().item())
        acc["log10"].append((p.log10() - g.log10()).abs().mean().item())
        ratio = torch.maximum(p / g, g / p)
        acc["delta_1_25"].append((ratio < 1.25).double().mean().item())
    out = {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc.items()}
    out["n_images"] = len(acc["abs_rel"])
    return out
