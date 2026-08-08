#!/usr/bin/env bash
# Does the rotary break VSSD at T=1025 because of angular SPREAD, or for some other reason?
#
# theta is a plain cumsum in the collapse, so its spread across the sequence grows linearly
# with T -- about 160 turns at T=1025 against 10 at T=65. VSSD pools the keys into one
# state before any query reads them, so those per-token rotations cannot cancel against a
# query the way they do in a pairwise score; they interfere, and a spread of many turns
# leaves the pooled state incoherent. Wrapping theta would prove nothing: cos/sin are
# 2pi-periodic, so only the spread matters.
#
# --rope-angle-scale 0.0634 = 65/1025 gives T=1025 the same total spread T=65 had. If the
# spread is the cause, this run trains; if it still sits at chance, the cause is elsewhere.
# 6 epochs is enough: the control reaches ~43% by epoch 5 while the collapsed run never
# leaves ~17%.
set -uo pipefail
cd "$(dirname "$0")/../.."
CFG=src/eval/configs/cifar10_patch1_rope.yaml

run() {
  echo "=== $2 ($(date -Is)) ==="
  uv run python -m eval.run_cifar --config "$CFG" --variants vit_mamba3_vssd \
      --no-rope --rope-angles --no-cudnn --epochs 6 --rope-angle-scale "$1" \
      --out "result/$2" 2>&1 | tee "result/$2/train.log" | grep -E "^ *ep "
}
mkdir -p result/rotary_spread_scale1 result/rotary_spread_scaled
run 1.0    rotary_spread_scale1    # control: reproduces the collapse
run 0.0634 rotary_spread_scaled    # same angular spread as T=65
echo "=== done ($(date -Is)) ==="
