#!/usr/bin/env bash
# Run the bidirectional depth retrain once the rope-turns grid is finished.
#
# The retrain was stopped 39 min into Phase B and requeued here: the n sweep matters more for
# the paper, and sharing one 12 GB GPU between them would slow both rather than help either.
#
# Gate on the grid's own completion marker first, then on the GPU actually being idle. Gating
# on the marker alone would race the last cell's teardown; gating on idleness alone would fire
# in the gap between two cells.
set -uo pipefail
cd "$(dirname "$0")/.."
for _ in $(seq 1 900); do                       # ceiling 15 h
  if grep -q "GRID DONE" result/turns_grid.log 2>/dev/null; then break; fi
  pgrep -f "sweep_turns_grid" >/dev/null || { echo "[queue] grid gone without GRID DONE"; break; }
  sleep 60
done
for _ in $(seq 1 60); do
  pgrep -f "\.venv/bin/python3 -m eval\.run_cifar" >/dev/null || break
  sleep 60
done
echo "[queue] starting bidirectional depth retrain at $(date -Is)"
MIXER=bidirectional bash scripts/run_depth_variant.sh
