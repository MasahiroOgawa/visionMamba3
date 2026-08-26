# shellcheck shell=bash
# Source before any GPU job in this repo: `source scripts/cudnn_env.sh`
#
# THE PROBLEM. torch ships its own cuDNN inside the venv, but that wheel does not carry every engine
# sub-library. cuDNN's dispatcher dlopens the missing one regardless, finds the host's copy under
# /usr/lib, and refuses the mixture with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH; every convolution
# then fails, which cudnn_guard.py works around by disabling cuDNN entirely -- correct but slow.
# Concretely on this host: torch bundles 9.20, which has no libcudnn_engines_tensor_ir, and the
# system ships 9.25 which does. /proc/self/maps showed five cuDNN libraries loaded from the venv and
# exactly one from /usr, and that one was the mismatch.
#
# THE FIX. Put a COMPLETE cuDNN matching the SYSTEM version ahead of the bundled one on the loader
# path, so every sub-library comes from one release. No sudo, no torch change, no lockfile change.
#
# WHY NOT UPGRADE THE VENV'S nvidia-cudnn-cu13 INSTEAD. torch pins an exact cuDNN, and asking for a
# newer one makes the resolver downgrade torch itself (2.12.1 -> 2.10.0 here) and swap the whole CUDA
# stack, which would invalidate every measurement. That was tried, and reverted.
#
# NOT A SPEED FIX. A correct stack is worth having and costs nothing, but it does not make these jobs
# fast: measured on WAFT track generation, 32.7 s/clip with cuDNN against 35.4 without, about 1.08x.
# An earlier claim of 15.8x was wrong -- it read tqdm's instantaneous rate over a stretch where most
# clips were already on disk and skipped without work.
#
# THE VERSION IS DETECTED, NOT HARDCODED. An apt upgrade that moves the system cuDNN silently
# reintroduces the mismatch under a hardcoded path, and the symptom is a slow job rather than an
# error, so it would go unnoticed. If no matching wheel is present this says so, prints the exact
# command to fetch one, and lets the job run on the no-cuDNN fallback.

_cudnn_setup() {
  local sys_lib sys_ver mm dir
  sys_lib=$(ls -1 /usr/lib/x86_64-linux-gnu/libcudnn.so.9.* 2>/dev/null | head -1)
  [ -n "$sys_lib" ] || return 0        # no system cuDNN: nothing can shadow the wheel

  sys_ver=${sys_lib##*libcudnn.so.}    # e.g. 9.25.0
  mm=${sys_ver%.*}                     # e.g. 9.25
  dir="$HOME/.local/lib/cudnn-${mm}/nvidia/cudnn/lib"

  if [ -d "$dir" ]; then
    export LD_LIBRARY_PATH="$dir:${LD_LIBRARY_PATH:-}"
    return 0
  fi

  {
    echo "[cudnn_env] system cuDNN is ${sys_ver}, but ~/.local/lib/cudnn-${mm} does not exist."
    echo "[cudnn_env] GPU jobs will run on the no-cuDNN fallback. To restore it, fetch a matching wheel:"
    echo "[cudnn_env]   uv run python - <<'PY'"
    echo "[cudnn_env]   import json, urllib.request"
    echo "[cudnn_env]   d = json.load(urllib.request.urlopen('https://pypi.org/pypi/nvidia-cudnn-cu13/json'))"
    echo "[cudnn_env]   v = max(r for r in d['releases'] if r.startswith('${mm}.') and 'dev' not in r)"
    echo "[cudnn_env]   f = json.load(urllib.request.urlopen(f'https://pypi.org/pypi/nvidia-cudnn-cu13/{v}/json'))"
    echo "[cudnn_env]   u = [x for x in f['urls'] if x['filename'].endswith('.whl') and 'x86_64' in x['filename']][0]"
    echo "[cudnn_env]   urllib.request.urlretrieve(u['url'], '/tmp/cudnn.whl')"
    echo "[cudnn_env]   PY"
    echo "[cudnn_env]   unzip -qo /tmp/cudnn.whl -d ~/.local/lib/cudnn-${mm}"
  } >&2
}
_cudnn_setup
unset -f _cudnn_setup
