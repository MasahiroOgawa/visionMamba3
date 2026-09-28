# visionMamba3

Vision Mamba-3: attention modules for vision built on the **State Space Duality (SSD)** of
Mamba-3, which was designed for text.

## 1. What is this repository

Mamba-3's SSD form writes a state-space layer as a masked attention

```
Y = (L ⊙ (C Bᵀ)) V
```

where `B`, `C` are key/query-like projections, `V` is the value projection, and `L` is a
**structured decay mask** built from per-token decay scalars rather than learned scores.
Because `L` is structured, the layer runs as a chunked scan in O(T·N) instead of softmax
attention's O(T²).

This repository turns that into drop-in `(B, T, C) → (B, T, C)` attention layers for image
tokens, and evaluates them:

| Module | What it is |
|---|---|
| `Mamba3SelfAttention` | Scan-based SSD self-attention. Bidirectional (2 scan directions) or 4-directional (adds column-major scans), Mamba-3 three-term (trapezoidal) or Mamba-2 two-term mask, optional fused Triton kernel from `mamba_ssm`. |
| `Mamba3VSSDAttention` | Non-causal SSD (NC-SSD, after VSSD, ICCV 2025): the mask collapses to a per-token scalar, so there is no scan order. |
| `Mamba3VSSDBetaGammaAttention` | NC-SSD with a second, independently-read pool (VSSD-β,γ). |
| `Mamba3CrossAttention` | SSD cross-attention between a query stream and a key/value stream, linear in T. |
| `RoPE2D` | 2-D rotary position embedding for image tokens, sharing its rotation-pair layout with Mamba-3's internal complex rotary. |

Two experiments use them:

- **CIFAR-10 operator comparison** (`src/eval`) — a small ViT whose mixer is swapped between
  softmax attention, 2-/4-directional SSD and NC-SSD, plus a CNN baseline, at a short (patch 4,
  T=65) and a long (patch 1, T=1025) sequence length. Reports accuracy, latency, peak memory and
  parameter count (Table 1 of the paper in `doc/attention`).
- **ETH3D depth** (`src/depth`) — a Depth-Anything-3 student whose attention is replaced by a
  Mamba-3 mixer: distilled from the DA3-SMALL teacher's features, then fine-tuned on ETH3D ground
  truth and evaluated on the held-out `terrains` scene.

The derivation and write-up live in `doc/` as LaTeX.

## 2. Structure of this repository

```
visionMamba3/
├── src/
│   ├── visionmamba3/          # the library: pure, reusable attention modules
│   │   ├── self_attention.py  #   Mamba3SelfAttention (scan SSD, reference + fused paths)
│   │   ├── vssd_attention.py  #   Mamba3VSSDAttention, Mamba3VSSDBetaGammaAttention (NC-SSD)
│   │   ├── cross_attention.py #   Mamba3CrossAttention
│   │   ├── mask.py            #   structured decay-mask builders (two-/three-term, cross)
│   │   ├── projections.py     #   AttentionProjections, BCNorm
│   │   └── rope2d.py          #   RoPE2D, rotate_pairs
│   ├── eval/                  # CIFAR-10 operator comparison
│   │   ├── run_cifar.py       #   entry point: train each variant, measure efficiency
│   │   ├── models.py          #   CNN / ViT-Tiny and the mixer swap (VARIANTS)
│   │   ├── train.py, data.py, efficiency.py
│   │   ├── configs/*.yaml     #   experiment recipes
│   │   └── run_*.sh           #   sweeps (Table 1, RoPE ablations, T=1025 grid, ...)
│   └── depth/                 # ETH3D depth experiment
│       ├── run.py             #   entry point: distill / finetune / eval
│       ├── student.py         #   DepthStudent (DA3 backbone with Mamba-3 mixers)
│       ├── da3.py             #   loading Depth-Anything-3 and driving its DPT head
│       ├── eth3d.py           #   ETH3D loader and the fixed scene split
│       └── metrics.py         #   depth metrics (median-aligned)
├── scripts/                   # job queues, sweeps, table/plot generation, cudnn_env.sh
├── tests/unit/                # pytest suite for the library and loaders
├── doc/
│   ├── attention/             # paper: mamba3_attention.tex (+ generated table/plot)
│   ├── mamba3_derivation/     # Mamba-3 derivation notes (.tex / .md)
│   └── original_paper/        # reference PDFs (Mamba 1-3, VSSD, Depth-Anything-3)
├── third_party/               # git submodules (official upstream, pinned)
│   ├── mamba-ssm/             #   state-spaces/mamba — Triton SSD kernels
│   └── depth-anything-3/      #   ByteDance-Seed/Depth-Anything-3 — teacher + DPT head
├── data/                      # datasets (not tracked)
├── result/, runs/             # experiment outputs (not tracked)
└── pyproject.toml, uv.lock
```

`third_party/` are submodules rather than installed packages because the code calls their
internals — the `mamba_ssm` Triton kernels decide the numbers, so the pinned commit is part of
every result. Only `mamba_ssm.ops.triton` is loaded; `visionmamba3/__init__.py` registers
`mamba_ssm` as a bare namespace package so its heavy top-level `__init__` (which needs
`tilelang`, `cutlass`, `quack`) never runs.

## 3. How to set up

Requirements: Linux, Python 3.11–3.12, [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU with
CUDA for the fused kernel and the experiments (the library itself also runs on CPU through the
PyTorch reference path).

```bash
git clone --recursive https://github.com/MasahiroOgawa/visionMamba3.git
cd visionMamba3
# if cloned without --recursive:
git submodule update --init --recursive

uv sync                          # library + dev tools
uv sync --extra eval             # + CIFAR-10 experiment (torchvision, pyyaml, matplotlib)
uv sync --extra eval --extra depth   # + ETH3D depth experiment (timm, opencv, safetensors, ...)
```

Before GPU jobs, source the cuDNN helper. If a system-wide cuDNN is installed and shadows the one
bundled with torch, convolutions fail with `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`; the script
detects the system version and puts a matching complete cuDNN first on the loader path (it prints
how to fetch one if it is missing):

```bash
source scripts/cudnn_env.sh
```

### Data

Nothing is downloaded automatically.

- **CIFAR-10** — the Python version under `data/cifar10/cifar-10-batches-py/`
  (unpack `cifar-10-python.tar.gz` from https://www.cs.toronto.edu/~kriz/cifar.html).
- **ETH3D** — the high-res multi-view DSLR scenes from https://www.eth3d.net/datasets, unpacked
  per scene under `data/eth3d/<scene>/` (each with `images/` and `ground_truth_depth/`).
  Training uses `courtyard, delivery_area, electro, facade, kicker, office, pipes, playground,
  relief, relief_2`; `terrains` is the test scene.
- The DA3-SMALL teacher (`depth-anything/DA3-SMALL`) is fetched from the Hugging Face Hub on first
  use.

### Check the install

```bash
uv run pytest
```

## 4. How to use

### As a library

The self-attention modules share one constructor and forward signature, so operators can be
swapped without changing call sites. `pos` is the `(B, T, 2)` integer `(y, x)` grid position of
each token, consumed by the optional `rope` module.

```python
import torch
from visionmamba3.self_attention import Mamba3SelfAttention
from visionmamba3.vssd_attention import Mamba3VSSDAttention
from visionmamba3.rope2d import RoPE2D

x = torch.randn(2, 16 * 16, 384, device="cuda")                  # (B, T, D)
ys, xs = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
pos = torch.stack([ys, xs], -1).reshape(1, -1, 2).expand(2, -1, -1).cuda()

attn = Mamba3SelfAttention(
    dim=384, num_heads=6, state_dim=64,
    rope=RoPE2D(), num_directions=2,   # 4 adds column-major scans
    row_renorm=False,                  # the fused kernel does not implement row renormalisation
).cuda()
y = attn(x, pos=pos, grid=(16, 16))    # (B, T, D)

nc = Mamba3VSSDAttention(dim=384, num_heads=6, state_dim=64, rope=RoPE2D()).cuda()
y = nc(x, pos=pos)
```

Cross-attention takes two token streams:

```python
from visionmamba3.cross_attention import Mamba3CrossAttention

xattn = Mamba3CrossAttention(dim_q=384, dim_kv=384, num_heads=6, state_dim=64)
y = xattn(q_tokens, kv_tokens)         # (B, T_q, dim_q)
```

Pass `use_fused_kernel=False` to `Mamba3SelfAttention` to use the PyTorch reference path (CPU or
kernel debugging).

### CIFAR-10 operator comparison

Each recipe is a YAML config in `src/eval/configs/`; the config's `variants` list chooses which
models to train (`cnn`, `vit_attn`, `vit_mamba3`, `vit_mamba3_4dir`, `vit_mamba3_vssd`,
`vit_mamba3_vssd_bg`) and `out` where results go.

```bash
source scripts/cudnn_env.sh
uv run python -m eval.run_cifar --config src/eval/configs/cifar10_patch4_rope.yaml   # T=65
uv run python -m eval.run_cifar --config src/eval/configs/cifar10_patch1_rope.yaml   # T=1025
```

Each run writes `<out>/results.json` with the resolved config and, per variant, accuracy,
latency, peak memory and parameter count. The full Table 1 (all five rows, both sequence lengths)
is:

```bash
bash src/eval/run_table1.sh          # -> result/table1_patch4, result/table1_patch1
```

The other `src/eval/run_*.sh` and `scripts/sweep_*.sh` scripts run the RoPE / rotary ablations.

### ETH3D depth

Three stages — feature distillation from the DA3 teacher, fine-tuning against ground truth, and
evaluation on `terrains`. `--mixer` is one of `bidirectional`, `vssd`, `vssd_bg`.

```bash
source scripts/cudnn_env.sh
uv run --extra depth python -m depth.run distill  --mixer vssd_bg --steps 20000 --out runs/vssd_bg
uv run --extra depth python -m depth.run finetune --mixer vssd_bg --init runs/vssd_bg/ckpt.pt \
    --steps 500 --lr-mixer 1e-4 --lr-bridge 3e-4 --out runs/vssd_bg_ft500
uv run --extra depth python -m depth.run finetune --mixer vssd_bg --init runs/vssd_bg_ft500/ckpt.pt \
    --steps 1000 --lr-mixer 1e-5 --lr-bridge 3e-5 --lr-head 1e-5 --unfreeze-head --augment \
    --out runs/vssd_bg_ft1000
uv run --extra depth python -m depth.run eval --mixer vssd_bg --ckpt runs/vssd_bg_ft1000/ckpt.pt
```

The two fine-tune stages are the reference recipe (head frozen, then unfrozen at a low rate);
one longer run is not equivalent. `scripts/run_depth_variant.sh` runs the whole chain for one
mixer (`MIXER=bidirectional bash scripts/run_depth_variant.sh`).

### Tables, figures and documents

```bash
uv run python scripts/make_ablation_table.py   # doc/attention/ablation_table.tex + ablation_mem_acc.png
make -C doc/attention                          # mamba3_attention.pdf
make -C doc/mamba3_derivation                  # mamba3_derivation.pdf
```

The table script reads every number from the run directories, so a cell with no run prints `--`.

## License

See [LICENSE](LICENSE).
