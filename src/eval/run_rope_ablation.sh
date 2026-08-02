#!/usr/bin/env bash
# Table 1 T=65 rows as a 2x2: {bidirectional SSD, NC-SSD} x {2-D RoPE, none}.
#
# Runs on the fused Triton kernel under bf16 autocast -- the fastest and most
# accurate path, and the same one the published rows used. The no-RoPE arm is
# the control: it should reproduce the published 83.7 / 75.1, which is what makes
# the RoPE cells trustworthy as a replacement for them.
#
# row_renorm is forced False for both arms (see build_model): the Triton kernel
# silently ignores it, so the constructor default of True would quietly make the
# reference path a different operator than the kernel one.
#
#   bash src/eval/run_rope_ablation.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

cfg=src/eval/configs/cifar10_patch4_rope.yaml

for arm in rope norope; do
  out="result/cifar10_patch4_$arm"
  flag=$([ "$arm" = rope ] && echo --rope || echo --no-rope)
  mkdir -p "$out"
  echo "=== T=65 $arm -> $out  ($(date -Is)) ==="
  uv run python -m eval.run_cifar --config "$cfg" "$flag" --out "$out" \
      2>&1 | tee "$out/train.log"
done
echo "=== all runs done ($(date -Is)) ==="
