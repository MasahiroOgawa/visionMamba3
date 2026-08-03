#!/usr/bin/env bash
# Second half of the Vision Mamba-3 grid: the rotary-ON cells.
#
# Full grid is {2-dir, 4-dir, NC-SSD} x {2-D RoPE on, off} x {rotary on, off}.
# run_vm3_ablation.sh covers the six rotary-off cells. Of the six rotary-on
# cells, two already exist from the Table 1 sweep (2-dir 70.51 and 4-dir 85.23,
# both with 2-D RoPE on), leaving four to run here:
#
#   * 2-D RoPE off + rotary on, all three operators
#   * 2-D RoPE on  + rotary on, NC-SSD only
#
# NC-SSD needs re-running because it silently ignored rope_angles until now, so
# its earlier "rotary on" number (78.91) actually measured the rotary-off cell.
#
#   bash src/eval/run_vm3_ablation_rotary.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

run() {  # run <out-suffix> <rope-flag> <variants...>
  local out="result/vm3_ablation_patch4_$1"; shift
  local rope_flag="$1"; shift
  mkdir -p "$out"
  echo "=== VM3 ablation T=65 $out (rotary on)  ($(date -Is)) ==="
  uv run python -m eval.run_cifar \
      --config src/eval/configs/cifar10_patch4_rope.yaml \
      --variants "$@" --rope-angles "$rope_flag" --out "$out" 2>&1 | tee "$out/train.log"
}

run norope_rotary --no-rope vit_mamba3 vit_mamba3_4dir vit_mamba3_vssd
run rope_rotary   --rope   vit_mamba3_vssd
echo "=== rotary-on cells complete ($(date -Is)) ==="
