#!/usr/bin/env bash
# One mixer through the full ETH3D depth chain, from this repo.
#
#   MIXER=bidirectional bash scripts/run_depth_variant.sh
#
# The three stages follow the recipe that produced the reference abs_rel 0.0531, recovered
# from that run's own logs rather than from prose:
#   Phase B  distil 20000 steps against DA3-SMALL's features, lr 3e-4
#   Phase C  500 steps, lr_mixer 1e-4 / lr_bridge 3e-4, head frozen, no augmentation
#   Phase C  1000 steps, head unfrozen at 1e-5, augmentation on
# The second Phase C starts from the first rather than replacing it: it begins from an
# already-adapted bridge, so one longer run is not the same experiment.
#
# n_views=1 matches the reference, which ran one image per step. It is also 2x faster than
# two views, and Phase-B distillation is per-token feature matching with no cross-view term
# to gain from more.
set -uo pipefail
cd "$(dirname "$0")/.."

MIXER=${MIXER:?set MIXER to bidirectional | vssd | vssd_bg}
STEPS_B=${STEPS_B:-20000}
NV=${NV:-1}
OUT=${OUT:-result/depth/$MIXER}
RUN="uv run --extra depth python -m depth.run"

mkdir -p "$OUT"

echo "=== [$MIXER] Phase B: distil ${STEPS_B} ($(date -Is)) ==="
$RUN distill --mixer "$MIXER" --steps "$STEPS_B" --n-views "$NV" \
     --lr-mixer 3e-4 --out "$OUT/distill" 2>&1 | grep -vE "^\[INFO|Warning:" \
  || { echo "[$MIXER] Phase B FAILED"; exit 1; }

echo "=== [$MIXER] Phase C 500, head frozen ($(date -Is)) ==="
$RUN finetune --mixer "$MIXER" --init "$OUT/distill/ckpt.pt" --steps 500 --n-views "$NV" \
     --lr-mixer 1e-4 --lr-bridge 3e-4 --out "$OUT/ft500" 2>&1 | grep -vE "^\[INFO|Warning:" \
  || { echo "[$MIXER] Phase C(500) FAILED"; exit 1; }

echo "=== [$MIXER] Phase C 1000, head trained ($(date -Is)) ==="
$RUN finetune --mixer "$MIXER" --init "$OUT/ft500/ckpt.pt" --steps 1000 --n-views "$NV" \
     --lr-mixer 1e-5 --lr-bridge 3e-5 --lr-head 1e-5 --unfreeze-head --augment \
     --out "$OUT/ft1000" 2>&1 | grep -vE "^\[INFO|Warning:" \
  || { echo "[$MIXER] Phase C(1000) FAILED"; exit 1; }

echo "=== [$MIXER] eval on held-out terrains ($(date -Is)) ==="
$RUN eval --mixer "$MIXER" --ckpt "$OUT/ft1000/ckpt.pt" \
     --out "$OUT/result.json" 2>&1 | grep -vE "^\[INFO|Warning:"

echo "=== [$MIXER] done ($(date -Is)) ==="
echo "  reference, same recipe in the retired repo: bidirectional 0.0531, VSSD-beta,gamma 0.0583"
echo "  DA3-SMALL zero-shot: 0.0417"
