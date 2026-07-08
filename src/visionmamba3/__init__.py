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
