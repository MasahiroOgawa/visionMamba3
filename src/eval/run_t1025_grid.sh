#!/usr/bin/env bash
# The 3x3 Vision Mamba-3 grid at T=1025: {2-dir, 4-dir, NC-SSD} x {no positional
# encoding, 2-D RoPE, 2-D RoPE + rotary}.
#
# Every existing T=1025 mamba number predates the RoPE2D pairing fix, and the
# both-encodings cells were measured while the two rotations acted in mismatched
# planes, so none of them are reusable. Flags are always explicit: the per-operator
# rotary default would otherwise silently differ between the three columns.
set -euo pipefail
cd "$(dirname "$0")/../.."

run() {  # run <out-suffix> <rope flag> <rotary flag>
  local out="result/vm3_t1025_$1"; shift
  mkdir -p "$out"
  echo "=== T=1025 $out  ($(date -Is)) ==="
  uv run python -m eval.run_cifar --config src/eval/configs/cifar10_patch1_rope.yaml \
      --variants vit_mamba3 vit_mamba3_4dir vit_mamba3_vssd \
      "$1" "$2" --out "$out" 2>&1 | tee "$out/train.log"
}

run norope      --no-rope --no-rope-angles
run rope        --rope    --no-rope-angles
run rope_rotary --rope    --rope-angles
echo "=== T=1025 grid complete ($(date -Is)) ==="
