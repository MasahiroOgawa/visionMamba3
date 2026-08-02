"""visionmamba3 — Mamba-3 SSD attention modules (pure vision library)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

_THIRD_PARTY = Path(__file__).resolve().parent.parent.parent / "third_party"

_MAMBA_SSM = _THIRD_PARTY / "mamba-ssm"
if _MAMBA_SSM.exists() and str(_MAMBA_SSM) not in sys.path:
    for _stub in ("selective_scan_cuda", "causal_conv1d_cuda"):
        if _stub not in sys.modules:
            sys.modules[_stub] = types.ModuleType(_stub)
    sys.path.insert(0, str(_MAMBA_SSM))
    # mamba_ssm/__init__.py eagerly imports its high-level models, which drag in
    # the cute and tilelang backends (needing tilelang, cutlass and quack). We only
    # ever use the Triton SSD kernels under mamba_ssm.ops.triton, and every
    # __init__ on the way down to them is empty -- so register mamba_ssm as a bare
    # namespace package over the real directory. Submodule imports then resolve
    # normally while the heavy top-level __init__ never runs. Without this the
    # fused SSD kernel is unreachable unless those unrelated backends are
    # installed, and Mamba3SelfAttention silently falls back to the slower,
    # numerically different reference path.
    if "mamba_ssm" not in sys.modules:
        _pkg = types.ModuleType("mamba_ssm")
        _pkg.__path__ = [str(_MAMBA_SSM / "mamba_ssm")]
        sys.modules["mamba_ssm"] = _pkg
