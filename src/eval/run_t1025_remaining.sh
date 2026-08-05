#!/usr/bin/env bash
# Complete Table 1's T=1025 half: the cells run_t1025_grid.sh does not cover.
#
# That grid runs {2-dir, 4-dir, VSSD-gamma} x {neither, 2-D RoPE, both}. Two gaps
# remain to make T=1025 the same 4x4 factorial as T=65:
#
#   1. the rotary-only column for those three operators, which is what separates the
#      rotary's own effect from its interaction with 2-D RoPE -- at T=65 the rotary
#      alone helped every operator (+1.15, +0.73, +3.39) while its marginal effect on
#      top of RoPE was negative for the scans, and only the rotary-only cell can tell
#      those apart;
#   2. VSSD-beta,gamma in all four encodings, since it postdates the grid.
#
# Flags are explicit everywhere: VSSD-gamma and VSSD-beta,gamma default the rotary
# ON per operator, so relying on defaults would silently make the "neither" and
# "2-D RoPE" columns rotary-on for two of the four operators and destroy the
# factorial this script exists to complete.
set -euo pipefail
cd "$(dirname "$0")/../.."

CFG=src/eval/configs/cifar10_patch1_rope.yaml

run() {  # run <out-suffix> <rope flag> <rotary flag> <variants...>
  local out="result/vm3_t1025_$1"; shift
  local rope="$1" rot="$2"; shift 2
  mkdir -p "$out"
  echo "=== T=1025 $out  [$*]  ($(date -Is)) ==="
  uv run python -m eval.run_cifar --config "$CFG" --variants "$@" \
      "$rope" "$rot" --out "$out" 2>&1 | tee "$out/train.log"
}

# 1. Rotary-only column for the three original operators.
run rotary --no-rope --rope-angles vit_mamba3 vit_mamba3_4dir vit_mamba3_vssd

# 2. VSSD-beta,gamma, all four encodings. One variant per directory so the table's
#    per-cell directory mapping stays one-to-one.
run vssdbg_norope      --no-rope --no-rope-angles vit_mamba3_vssd_bg
run vssdbg_rotary      --no-rope --rope-angles    vit_mamba3_vssd_bg
run vssdbg_rope        --rope    --no-rope-angles vit_mamba3_vssd_bg
run vssdbg_rope_rotary --rope    --rope-angles    vit_mamba3_vssd_bg

echo "=== T=1025 remaining cells complete ($(date -Is)) ==="
