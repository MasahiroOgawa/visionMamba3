#!/usr/bin/env bash
# The three VSSD-beta,gamma cells of Table 1 at T=65: no positional encoding,
# 2-D RoPE, and 2-D RoPE + Mamba-3's complex rotary.
#
# Same recipe and config as every other T=65 row, so the new rows are comparable
# with the existing nine on everything except parameter count -- VSSD-beta,gamma
# carries a second query and scalar projection, 2.93 M against 2.71 M, which is
# inherent to the operator and visible in the table's Params column.
#
# Both flags are explicit in every call: this variant's per-operator rotary default
# is on (it inherits VSSD-gamma's reason -- a per-token scalar mask with no |i-j|
# term), so relying on the default would silently make the first two columns
# rotary-on and destroy the comparison the grid exists to make.
set -euo pipefail
cd "$(dirname "$0")/../.."

run() {  # run <out-suffix> <rope flag> <rotary flag>
  local out="result/vm3_vssdbg_$1"; shift
  mkdir -p "$out"
  echo "=== VSSD-beta,gamma $out  ($(date -Is)) ==="
  uv run python -m eval.run_cifar --config src/eval/configs/cifar10_patch4_rope.yaml \
      --variants vit_mamba3_vssd_bg \
      "$1" "$2" --out "$out" 2>&1 | tee "$out/train.log"
}

run norope      --no-rope --no-rope-angles
run rope        --rope    --no-rope-angles
run rope_rotary --rope    --rope-angles
echo "=== VSSD-beta,gamma T=65 sweep complete ($(date -Is)) ==="
