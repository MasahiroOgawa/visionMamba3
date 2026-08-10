#!/usr/bin/env bash
# Run VSSD-gamma once the GPU is free. Gate on "no depth.run process alive" rather than on a
# result file: if an earlier arm dies, its result.json never appears and a file gate would
# wait forever, whereas this one proceeds as soon as the GPU is actually idle.
set -uo pipefail
cd "$(dirname "$0")/.."
for _ in $(seq 1 720); do                       # ceiling 12 h
  pgrep -f "\.venv/bin/python3? -m depth\.run" >/dev/null || break
  sleep 60
done
if pgrep -f "\.venv/bin/python3? -m depth\.run" >/dev/null; then
  echo "[queue] GPU still busy after 12 h; not starting VSSD-gamma"; exit 1
fi
echo "[queue] GPU free at $(date -Is); starting VSSD-gamma"
MIXER=vssd bash scripts/run_depth_variant.sh
