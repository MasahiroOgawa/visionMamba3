#!/usr/bin/env bash
# Re-measure the three RoPE-only cells under the corrected RoPE2D pairing.
#
# Those cells (2-dir 82.71, 4-dir 83.26, NC-SSD 79.40) were measured while
# RoPE2D half-split its channel pairs. RoPE alone is *functionally* unaffected by
# the pairing -- it is a relabelling of learned channels, and the relative-position
# check is clean either way -- but not bit-identical, so re-measuring is what makes
# the published grid one consistent experiment. This also gives a second
# measurement of 4-dir + RoPE, the current best configuration, which was n=1.
#
# --no-rope-angles is explicit: NC-SSD's per-operator default is now rotary-on,
# and this sweep is the rotary-off column.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=result/vm3_ropeonly_recheck
mkdir -p "$out"
uv run python -m eval.run_cifar --config src/eval/configs/cifar10_patch4_rope.yaml \
    --variants vit_mamba3 vit_mamba3_4dir vit_mamba3_vssd \
    --rope --no-rope-angles --out "$out" 2>&1 | tee "$out/train.log"
echo "=== done ($(date -Is)) ==="
