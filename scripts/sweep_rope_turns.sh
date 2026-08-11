#!/usr/bin/env bash
# Sweep the rotary's pinned angular spread n at T=1025, on the SAME protocol as Table 1.
#
#   bash scripts/sweep_rope_turns.sh
#
# Why this exists: an earlier spread sweep (result/rotary_scale_*) used --rope-angle-scale and
# ran only 6 epochs, so it topped out near 52% against Table 1's 59.0 for n=1 and could not be
# quoted beside it. Same build (2.92 M at T=1025, i.e. 2.74 M plus the larger pos-embed) --
# the difference was purely training length. These runs use the full 30-epoch protocol, so each
# point is directly comparable with the existing n=1 row.
#
# n=1 is NOT re-run: result/vm3_t1025_turns1_vssd already has it at 58.98 under this protocol.
# n<1 is included because "n=1 is best" cannot be claimed without a point below it.
set -uo pipefail
cd "$(dirname "$0")/.."
# n=1 IS re-run here even though result/vm3_t1025_turns1_vssd already has it at 58.98: that run
# predates the cuDNN breakage and so used cuDNN for the patch-embed conv, while these must pass
# --no-cudnn. Re-running it removes the last uncontrolled difference between the points.
TURNS=${TURNS:-"0.5 1 2 3 10"}
VARIANT=${VARIANT:-vit_mamba3_vssd}

for N in $TURNS; do
  OUT="result/vm3_t1025_turns${N}_vssd"
  # Skip only COMPLETED runs. A bare directory test would skip a crashed run's retry, which is
  # exactly what happened when the first attempt died on cuDNN in four seconds.
  if [ "$(grep -c '^  ep' "$OUT/train.log" 2>/dev/null || echo 0)" -ge 30 ]; then
    echo "[n=$N] SKIP: $OUT already has 30 epochs"; continue
  fi
  rm -rf "$OUT"
  echo "=== [n=$N] $VARIANT, T=1025, 30 epochs ($(date -Is)) ==="
  uv run python -m eval.run_cifar --variants "$VARIANT" --patch-size 1 --epochs 30 \
      --no-rope --no-cudnn --rope-turns "$N" --out "$OUT" 2>&1 | grep -vE "^\[INFO|Warning:" \
    || echo "[n=$N] FAILED"
  echo "=== [n=$N] done ($(date -Is)) ==="
done
echo "###### SWEEP DONE ######"
echo "  reference on this protocol: n=1 -> 58.98 ; rotary off -> 63.9 (Table 1)"
