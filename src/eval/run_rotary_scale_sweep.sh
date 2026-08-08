#!/usr/bin/env bash
# How much angular spread does VSSD's rotary actually want at T=1025?
#
# 1.0 (~160 turns) collapses to chance; 0.0634 (~10 turns, the spread T=65 had) trains and
# beats no-encoding. Both endpoints are known, so this fills in between and below to find
# whether there is an optimum, and specifically whether one single turn -- the smallest
# spread that still distinguishes the two ends of the sequence -- is better or worse.
#
# The trade-off the sweep is testing: spread = T * increment, while the resolution between
# neighbouring tokens is the increment itself. Shrinking the spread to keep the pooled sum
# coherent also shrinks how much neighbouring tokens differ, so too little spread should
# degenerate toward no encoding at all.
#
#   0.0063  ~1 turn over the sequence
#   0.0200  ~3 turns
#   0.2500  ~40 turns
# (0.0634 ~10 turns and 1.0 ~160 turns already measured)
set -uo pipefail
cd "$(dirname "$0")/../.."
CFG=src/eval/configs/cifar10_patch1_rope.yaml

for s in 0.0063 0.0200 0.2500; do
  out="result/rotary_scale_$s"
  mkdir -p "$out"
  echo "=== scale $s ($(date -Is)) ==="
  uv run python -m eval.run_cifar --config "$CFG" --variants vit_mamba3_vssd \
      --no-rope --rope-angles --no-cudnn --epochs 6 --rope-angle-scale "$s" \
      --out "$out" 2>&1 | tee "$out/train.log" | grep -E "^ *ep "
done
echo "=== sweep done ($(date -Is)) ==="
