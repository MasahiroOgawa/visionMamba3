#!/usr/bin/env bash
# Find the best pinned rotary spread n, for both collapse operators, on Table 1's protocol.
#
#   bash scripts/sweep_turns_grid.sh
#
# Grid: n in {0.25, 0.5, 1, 2, 4} x {VSSD-gamma, VSSD-beta,gamma}, T=1025, 30 epochs, no 2-D
# RoPE. Three cells already exist and are skipped: VSSD-gamma at 0.5 (64.67) and 1 (58.98),
# VSSD-beta,gamma at 1 (59.69).
#
# Completion is judged by results.json, not by train.log: the n=0.5 run wrote no train.log
# because its per-epoch output went to the driver's stdout, and a train.log test would have
# re-run a finished cell. A bare directory test is also wrong -- it silently skips the retry
# of a crashed run, which is how the first attempt at this sweep lost four runs.
#
# --no-cudnn is required on this host (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH on any conv2d).
set -uo pipefail
cd "$(dirname "$0")/.."

for _ in $(seq 1 720); do            # wait for the GPU, ceiling 12 h
  pgrep -f "\.venv/bin/python3 -m (depth\.run|eval\.run_cifar)" >/dev/null || break
  sleep 60
done
echo "[grid] GPU free at $(date -Is)"

for VAR in vit_mamba3_vssd vit_mamba3_vssd_bg; do
  SUF=$([ "$VAR" = vit_mamba3_vssd ] && echo vssd || echo vssdbg)
  for N in 0.25 0.5 1 2 4; do
    OUT="result/vm3_t1025_turns${N}_${SUF}"
    if [ -f "$OUT/results.json" ]; then echo "[$SUF n=$N] SKIP: already complete"; continue; fi
    rm -rf "$OUT"
    echo "=== [$SUF n=$N] 30 epochs ($(date -Is)) ==="
    uv run python -m eval.run_cifar --variants "$VAR" --patch-size 1 --epochs 30 \
        --no-rope --no-cudnn --rope-turns "$N" --out "$OUT" 2>&1 \
      | grep -vE "^\[INFO|Warning:" || echo "[$SUF n=$N] FAILED"
    echo "=== [$SUF n=$N] done ($(date -Is)) ==="
  done
done
echo "###### GRID DONE ######"
