#!/usr/bin/env bash
# Regenerate result/depth/spread.json for the paper's Table A.3.
#
# Table A.3 does NOT use result.json. Every row in it comes from measure_depth_spread.py, which
# re-scores all arms through one evaluator and also reports the per-view standard deviation the
# caption quotes. The two paths disagree substantially -- bidirectional is 0.0550 in result.json
# and 0.0791 in spread.json -- so the numbers must never be mixed between them.
#
# The spread.json on disk was written 2026-08-13 09:20, before all three current checkpoints
# (bidirectional 08-14 19:55, vssd 08-15 13:52, vssd_bg 08-15 16:30), so it describes superseded
# runs. Table A.3 cannot be updated until this re-run lands.
set -uo pipefail
cd "$(dirname "$0")/.."

for _ in $(seq 1 1440); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] && break
  sleep 60
done
echo "[rescore] GPU idle at $(date -Is)"

prev=result/depth/spread.json
[ -f "$prev" ] && cp "$prev" "result/depth/_spread_pre_$(date +%Y%m%d).json"

uv run python scripts/measure_depth_spread.py || { echo "[rescore] FAILED"; exit 1; }
python3 -c "
import json
for a in json.load(open('result/depth/spread.json')):
    print(f\"  {a['arm']:18s} abs_rel={a['abs_rel']:.4f}+-{a['abs_rel_sd']:.4f} rmse={a['rmse']:.4f} log10={a['log10']:.4f} d1.25={a['delta_1_25']:.4f}\")"
echo "[rescore] done at $(date -Is)"
