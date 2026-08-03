#!/usr/bin/env bash
# Table 1, all five rows, on the fully updated Vision Mamba-3 operator:
# fused Triton kernel + bf16 + external 2-D RoPE + Mamba-3's internal complex
# rotary, with the 2- and 4-directional SSD reported as separate rows so the
# column-major pair's contribution is visible (NC-SSD has no scan order at all --
# its mask collapses).
#
# Both sequence lengths, one self-consistent environment, so the mamba rows are
# comparable to the softmax and CNN rows measured alongside them rather than to
# numbers from a retired repo and an older kernel.
#
#   bash src/eval/run_table1.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

for tag in patch4 patch1; do
  out="result/table1_$tag"
  mkdir -p "$out"
  echo "=== Table 1 $tag -> $out  ($(date -Is)) ==="
  uv run python -m eval.run_cifar \
      --config "src/eval/configs/cifar10_${tag}_rope.yaml" \
      --variants cnn vit_attn vit_mamba3 vit_mamba3_4dir vit_mamba3_vssd \
      --out "$out" 2>&1 | tee "$out/train.log"
done
echo "=== Table 1 complete ($(date -Is)) ==="
