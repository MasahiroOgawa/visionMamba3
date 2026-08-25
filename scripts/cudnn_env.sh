# shellcheck shell=bash
# Source before any GPU job in this repo: `source scripts/cudnn_env.sh`
#
# torch 2.12.1+cu130 bundles cuDNN 9.20, which does not ship libcudnn_engines_tensor_ir. cuDNN's
# dispatcher dlopens that engine anyway and finds the host's 9.25 copy in /usr/lib, then rejects it
# with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH. Every convolution then fails, which is what
# cudnn_guard.py works around by disabling cuDNN entirely -- correct but slow.
#
# CORRECTION: this restores cuDNN but does NOT speed up WAFT track generation. Measured on the
# same clips, 32.7 s/clip with cuDNN against 35.4 without -- about 1.08x. An earlier claim of
# 15.8x here was wrong: it compared tqdm's instantaneous rate over a stretch where most clips
# were already on disk and skipped instantly. WAFT's cost is dominated by something other than
# cuDNN convolutions. Keep this file because a correct cuDNN is worth having and costs nothing,
# not because it makes this job fast.
#
# The fix is to make the whole set 9.25, matching the host, by putting a complete 9.25 wheel ahead
# of the bundled one on the loader path. Upgrading the venv's nvidia-cudnn-cu13 instead does NOT
# work: torch 2.12.1 pins 9.20.0.48, and asking for 9.25 makes the resolver downgrade torch to
# 2.10.0, which would invalidate every measurement in the paper.
#
# Populate the directory with:
#   uv run python -c "import urllib.request,json; d=json.load(urllib.request.urlopen(
#     'https://pypi.org/pypi/nvidia-cudnn-cu13/9.25.0.15/json'));
#     u=[f for f in d['urls'] if f['filename'].endswith('.whl') and 'x86_64' in f['filename']][0];
#     urllib.request.urlretrieve(u['url'],'/tmp/cudnn.whl')"
#   unzip -qo /tmp/cudnn.whl -d ~/.local/lib/cudnn-9.25
CUDNN_925="$HOME/.local/lib/cudnn-9.25/nvidia/cudnn/lib"
if [ -d "$CUDNN_925" ]; then
  export LD_LIBRARY_PATH="$CUDNN_925:${LD_LIBRARY_PATH:-}"
else
  echo "[cudnn_env] $CUDNN_925 missing; jobs will fall back to the no-cuDNN path and run slowly" >&2
fi
