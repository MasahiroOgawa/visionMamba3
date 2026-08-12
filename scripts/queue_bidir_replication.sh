#!/usr/bin/env bash
# Attempt to replicate the reference implementation's 0.0531 with this repository's code,
# after the rope-turns grid frees the GPU.
#
# What is already established, so it is not re-tested here:
#   - Inference is correct. The reference's own CM22 weights, loaded into this pipeline and
#     scored on the reference path, give 0.0530 against its published 0.0531.
#   - Ruled out by measurement: loader, metric, architecture and parameter names, teacher
#     feature route, patch/positional embedding, patch_start_idx, view arrangement,
#     ref_view_strategy, and the fused Triton kernel.
#   - So the residual is in training. Our runs: 0.0695 (kernel-trained), 0.0833 (reference
#     path), against the reference's 0.0530.
#
# The one training difference found since: CosineAnnealingLR eta_min. The reference floors
# both phases at a tenth of the peak learning rate; ours annealed to exactly zero, so the
# final steps of each phase trained at no learning rate. Now fixed in depth/run.py.
set -uo pipefail
cd "$(dirname "$0")/.."
for _ in $(seq 1 900); do
  pgrep -f "\.venv/bin/python3 -m (eval\.run_cifar|depth\.run)" >/dev/null || break
  pgrep -f "sweep_turns_grid" >/dev/null || { sleep 30; }
  sleep 60
done
echo "[replication] GPU free at $(date -Is); running bidirectional with the LR floor"
mv result/depth/bidirectional result/depth/_bidirectional_no_lr_floor 2>/dev/null
MIXER=bidirectional bash scripts/run_depth_variant.sh
