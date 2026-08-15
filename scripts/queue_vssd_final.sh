#!/usr/bin/env bash
# Run VSSD-gamma and VSSD-beta,gamma on the recipe that reproduced the reference bidirectional
# result (abs_rel 0.0550 against the reference's 0.0531, 2026-08-14), so the four ETH3D rows --
# DA3-SMALL, bidirectional, VSSD-gamma, VSSD-beta,gamma -- all come from one configuration.
#
# "The recipe" is simply the current state of run_depth_variant.sh and src/depth/, which now
# carries all five corrections found while chasing the reference:
#   1. --augment passed to the final Phase-C stage (it never was)
#   2. the reference's augmentation: crop 0.6-1.0, hflip p=0.5, brightness/contrast/saturation
#      0.6-1.4, hue +-0.1 (ours had crop only, 0.7-1.0)
#   3. all ten training scenes (relief_2 and electro were withheld for a split nothing read)
#   4. all 374 images (load_scene used to take sorted(paths)[:1] -- ten images in total)
#   5. uniform sampling over the flat image list (scene-then-image skewed it 5.4x)
# Nothing is passed here that the bidirectional run did not get; the mixer is the only difference.
#
# Supersedes queue_vssd_reruns.sh, whose completion guard tested results.json while depth.run
# writes result.json, so it would have re-run a finished arm instead of skipping it.
set -uo pipefail
cd "$(dirname "$0")/.."

# Wait for the GPU, including the tracker training in the sibling repository. Match the venv
# interpreter rather than a bare command substring, so this loop cannot match its own wrapper.
for _ in $(seq 1 1440); do
  pgrep -f "\.venv/bin/python3? -m (depth\.run|eval\.run_cifar)" >/dev/null && { sleep 60; continue; }
  pgrep -f "\.venv/bin/python3? .*(train_depth_refined_tracker|eval_metric3d)" >/dev/null && { sleep 60; continue; }
  break
done
echo "[vssd-final] GPU free at $(date -Is)"

for MIXER in vssd vssd_bg; do
  OUT="result/depth/${MIXER}"
  # Completion guard on the file depth.run actually writes.
  if [ -f "$OUT/result.json" ] && [ -f "$OUT/.final_recipe" ]; then
    echo "[vssd-final] ${MIXER} already run on this recipe; skipping"; continue
  fi
  # Retire the previous result under a name that says what made it stale, and never overwrite an
  # existing archive.
  if [ -d "$OUT" ]; then
    dst="result/depth/_${MIXER}_pre_sampling_fix"
    [ -e "$dst" ] && dst="${dst}_$(date +%H%M%S)"
    mv "$OUT" "$dst" && echo "[vssd-final] retired previous ${MIXER} -> $(basename "$dst")"
  fi
  echo "=== [${MIXER}] start $(date -Is) ==="
  MIXER="${MIXER}" bash scripts/run_depth_variant.sh || { echo "[vssd-final] ${MIXER} FAILED"; continue; }
  touch "$OUT/.final_recipe"
  echo "=== [${MIXER}] done $(date -Is) ==="
done

echo "[vssd-final] all arms finished at $(date -Is)"
echo "  for the four-row table:"
for m in bidirectional vssd vssd_bg; do
  f="result/depth/${m}/result.json"
  [ -f "$f" ] && python3 -c "
import json,sys; d=json.load(open('$f'))
print(f\"    {'$m':14s} abs_rel={d['abs_rel']:.4f} rmse={d['rmse']:.4f} log10={d['log10']:.4f} d1.25={d['delta_1_25']:.4f}\")"
done
echo "    DA3-SMALL      abs_rel=0.0323 rmse=0.0605 log10=0.0141 d1.25=1.0000  (zero-shot, native two-stream path)"
