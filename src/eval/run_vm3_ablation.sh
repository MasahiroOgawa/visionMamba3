#!/usr/bin/env bash
# Vision Mamba-3 ablation at T=65: {2-dir, 4-dir, NC-SSD} x {2-D RoPE on, off}.
#
# Mamba-3's internal complex rotary is OFF in all six cells. The previous sweep
# turned it on at the same time as the 4-directional scans, and comparing its
# 2-directional row against the earlier no-rotary run suggests the rotary *costs*
# accuracy (T=65 82.71 -> 70.51, T=1025 77.89 -> 73.15). With both changed at
# once that attribution was unproven, so this grid holds the rotary off and
# varies only the external 2-D RoPE; the rotary-on cells from the previous sweep
# are the comparison points.
#
#   bash src/eval/run_vm3_ablation.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

for arm in rope norope; do
  out="result/vm3_ablation_patch4_$arm"
  flag=$([ "$arm" = rope ] && echo --rope || echo --no-rope)
  mkdir -p "$out"
  echo "=== VM3 ablation T=65 $arm (no rotary) -> $out  ($(date -Is)) ==="
  uv run python -m eval.run_cifar \
      --config src/eval/configs/cifar10_patch4_rope.yaml \
      --variants vit_mamba3 vit_mamba3_4dir vit_mamba3_vssd \
      --no-rope-angles "$flag" --out "$out" 2>&1 | tee "$out/train.log"
done
echo "=== ablation complete ($(date -Is)) ==="
