#!/usr/bin/env bash
# Fill the four unmeasured T=65 cells of the CIFAR-10 table: the pinned 1-turn rotary rows for
# VSSD-1pool and VSSD-2pool, with and without 2-D RoPE. Every other knob matches the existing
# T=65 rows exactly (patch 4, 80 epochs, batch 128, lr 3e-4, wd 0.05, seed 42, warmup 10,
# grad-clip 1.0, plateau), so the new cells are comparable to the ones already in the table.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"   # torch bundles cuDNN 9.20; the host ships 9.25 engines
COMMON=(--patch-size 4 --epochs 80 --batch-size 128 --lr 3e-4 --seed 42 \
        --warmup-epochs 10 --grad-clip 1.0 --lr-schedule plateau --rope-turns 1 --rope-angles)
run () {  # name variant rope-flag
  local out="result/$1"
  [ -f "$out/results.json" ] && { echo "[t65] $1 done"; return 0; }
  echo "=== [t65] $1 ($(date -Is)) ==="
  uv run python -m eval.run_cifar --variants "$2" "$3" --out "$out" "${COMMON[@]}" \
    || { echo "[t65] $1 FAILED"; return 1; }
}
run vm3_t65_turns1_vssd            vit_mamba3_vssd     --no-rope
run vm3_t65_turns1_rope_vssd       vit_mamba3_vssd     --rope
run vm3_t65_turns1_vssdbg          vit_mamba3_vssd_bg  --no-rope
run vm3_t65_turns1_rope_vssdbg     vit_mamba3_vssd_bg  --rope
echo "[t65] done at $(date -Is)"
