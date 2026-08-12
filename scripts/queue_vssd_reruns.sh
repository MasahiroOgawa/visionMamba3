#!/usr/bin/env bash
# Re-run both VSSD arms of the ETH3D study on the CURRENT code, once the bidirectional
# replication frees the GPU.
#
# Why these two must be re-run at all, independent of what bidirectional shows:
#   Appendix A.5's table may list only numbers this repository can reproduce. The existing
#   VSSD-gamma (0.0738) and VSSD-beta,gamma (0.0663) were both trained BEFORE
#   CosineAnnealingLR gained eta_min = lr * 0.1 in src/depth/run.py. That floor changes every
#   arm -- the VSSD arms ignore the Triton kernel, but not the learning-rate schedule -- so
#   neither number would come out the same today. Re-running is required by the
#   reproducibility rule, not conditional on the bidirectional attempt succeeding.
#
# The old results are moved aside rather than deleted, so the pre-floor numbers stay
# recoverable for comparison.
set -uo pipefail
cd "$(dirname "$0")/.."

# Wait for the GPU. Guard against matching this script's own pgrep (a bare command-line
# substring match picks up the wrapper shell too), so match the venv interpreter explicitly.
for _ in $(seq 1 900); do
  pgrep -f "\.venv/bin/python3 -m (eval\.run_cifar|depth\.run)" >/dev/null || break
  sleep 60
done
echo "[vssd-rerun] GPU free at $(date -Is)"

# Names as run_depth_variant.sh accepts them: vssd = VSSD-gamma, vssd_bg = VSSD-beta,gamma.
for MIXER in vssd vssd_bg; do
  # Retire the stale pre-floor result FIRST. Order matters: result/depth/$MIXER/results.json
  # already exists from the pre-floor run, so testing it before the move would skip the very
  # re-run this script exists to perform. The _no_lr_floor directory's absence is what marks
  # "not yet retired", and it is never overwritten, so a re-invocation cannot clobber it.
  if [ ! -d "result/depth/_${MIXER}_no_lr_floor" ]; then
    mv "result/depth/${MIXER}" "result/depth/_${MIXER}_no_lr_floor" 2>/dev/null
  fi
  # Completion-based skip, not directory-existence: a crashed run leaves the directory behind
  # but no results.json, and an existence guard silently skipped four retries once already.
  if [ -f "result/depth/${MIXER}/results.json" ]; then
    echo "[vssd-rerun] ${MIXER} already complete on current code; skipping"
    continue
  fi
  echo "=== [${MIXER}] start $(date -Is) ==="
  MIXER="${MIXER}" bash scripts/run_depth_variant.sh
  echo "=== [${MIXER}] done $(date -Is) ==="
done
echo "[vssd-rerun] all arms finished at $(date -Is)"
