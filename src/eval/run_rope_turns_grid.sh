#!/usr/bin/env bash
# The fixed-turn rotary on both VSSD operators, at T=1025, for Table 1.
#
# The learned cumulative angle spans ~160 turns at T=1025 and collapses both VSSD
# operators to chance; the spread sweep is monotone in favour of less spread, with one
# turn best (51.9% test at 6 epochs against 42.7% at ten turns and 14.6% at ~160).
# --rope-turns 1 pins theta_j = j*2*pi/T, so the spread is one turn whatever T is.
#
# Four cells, full 30 epochs so they are directly comparable with the rest of the grid:
# {VSSD-gamma, VSSD-beta,gamma} x {1-turn rotary alone, 1-turn rotary + 2-D RoPE}.
# The scan operators are left out deliberately -- their pairwise score already cancels the
# absolute angle, so they neither suffer the collapse nor need the fix.
set -uo pipefail
cd "$(dirname "$0")/../.."
CFG=src/eval/configs/cifar10_patch1_rope.yaml

run() {   # run <variant> <rope-flag> <tag>
  local out="result/$3"
  mkdir -p "$out"
  echo "=== $3 ($(date -Is)) ==="
  uv run python -m eval.run_cifar --config "$CFG" --variants "$1" \
      $2 --rope-angles --rope-turns 1 --no-cudnn \
      --out "$out" 2>&1 | tee "$out/train.log" | grep -E "^ *ep "
}

run vit_mamba3_vssd    --no-rope vm3_t1025_turns1_vssd
run vit_mamba3_vssd    --rope    vm3_t1025_turns1_rope_vssd
run vit_mamba3_vssd_bg --no-rope vm3_t1025_turns1_vssdbg
run vit_mamba3_vssd_bg --rope    vm3_t1025_turns1_rope_vssdbg
echo "=== grid done ($(date -Is)) ==="
