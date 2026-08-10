#!/usr/bin/env bash
# Extend Phase C by 9000 steps for all three mixers, continuing from each arm's existing
# 1000-step checkpoint (so the final stage totals 10000 steps).
#
#   bash scripts/extend_phase_c.sh
#
# Every checkpoint along the way is scored, not just the last one: depth error on the
# held-out scene is not monotone in the training loss, and an earlier long run in this
# project degraded while its loss kept falling. Picking the best-scoring step afterwards is
# only possible if the intermediate checkpoints exist.
#
# Settings are the ones the 1000-step stage used, so this is a continuation and not a new
# recipe: lr_mixer 1e-5, lr_bridge 3e-5, lr_head 1e-5, head unfrozen, SILog + 0.1*edge,
# no augmentation. CosineAnnealingLR restarts over the 9000 steps (a warm restart from 1e-5).
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="uv run --extra depth python -m depth.run"
STEPS=${STEPS:-9000}
EVERY=${EVERY:-3000}

for MIXER in bidirectional vssd_bg vssd; do
  SRC="result/depth/$MIXER/ft1000/ckpt.pt"
  OUT="result/depth/$MIXER/ft10000"
  if [ ! -f "$SRC" ]; then echo "[$MIXER] SKIP: $SRC missing"; continue; fi
  mkdir -p "$OUT"
  echo "=== [$MIXER] Phase C +$STEPS from ft1000 ($(date -Is)) ==="
  $RUN finetune --mixer "$MIXER" --init "$SRC" --steps "$STEPS" --n-views 1 \
       --lr-mixer 1e-5 --lr-bridge 3e-5 --lr-head 1e-5 --unfreeze-head \
       --ckpt-every "$EVERY" --out "$OUT" 2>&1 | grep -vE "^\[INFO|Warning:" \
    || { echo "[$MIXER] FAILED"; continue; }

  for CK in "$OUT"/ckpt_*.pt "$OUT/ckpt.pt"; do
    [ -f "$CK" ] || continue
    echo "--- [$MIXER] eval $(basename "$CK") ($(date -Is)) ---"
    $RUN eval --mixer "$MIXER" --ckpt "$CK" \
         --out "$OUT/result_$(basename "$CK" .pt).json" 2>&1 | grep -vE "^\[INFO|Warning:"
  done
  echo "=== [$MIXER] done ($(date -Is)) ==="
done
echo "###### ALL ARMS EXTENDED ######"
echo "  1500-step baselines: bidirectional 0.0695, VSSD-beta,gamma 0.0663, VSSD-gamma 0.0738"
echo "  DA3-SMALL zero-shot, matched path: 0.0323"
