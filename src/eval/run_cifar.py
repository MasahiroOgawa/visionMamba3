"""CIFAR-10 operator comparison — train a variant and measure its efficiency.

Fills the vmamba3 paper's Table 1 by running the NC-SSD (``vit_mamba3_vssd``)
variant at the long sequence length (patch_size=1 -> T=1025) under the exact
recipe that produced the already-published softmax / bidirectional rows.

  uv run python -m eval.run_cifar --variant vit_mamba3_vssd --patch-size 1 \
      --batch-size 32 --epochs 30 --lr 3e-4 --lr-schedule plateau \
      --warmup-epochs 10 --grad-clip 1.0 --seed 42 --out result/cifar10_patch1_vssd

Efficiency (latency/peak-mem) is probed at ``--eff-batch`` (default 128, the
batch used for the existing Table 1 rows) so NC-SSD is directly comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from eval.data import CIFAR10_TRAIN_N, make_loaders
from eval.efficiency import count_params, measure
from eval.models import VARIANTS, build_model
from eval.train import (
    WarmupCosineStrategy,
    WarmupPlateauStrategy,
    evaluate,
    set_seed,
    train_one_epoch,
)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(line_buffering=True)

_PATH_ARGS = ("out", "data_dir", "config")


def build_lr_strategy(args, optimizer, total_steps: int, warmup_steps: int):
    if args.lr_schedule == "plateau":
        return WarmupPlateauStrategy(
            optimizer, args.lr, warmup_steps,
            factor=args.plateau_factor, patience=args.plateau_patience,
            threshold=args.plateau_threshold, min_lr=args.plateau_min_lr,
            ema_alpha=args.plateau_loss_ema_alpha,
        )
    return WarmupCosineStrategy(optimizer, total_steps, warmup_steps)


def run_variant(variant: str, args, device: torch.device) -> dict:
    print(f"\n=== variant: {variant} ===")
    set_seed(args.seed)
    # Loaders are rebuilt per variant, *after* re-seeding, so every variant sees
    # the same shuffle order. Sharing one DataLoader across the sweep let its
    # generator state carry over, which made a variant's result depend on how
    # many variants ran before it -- NC-SSD moved 79.46 -> 78.91 purely from
    # inserting another variant ahead of it.
    train_dl, test_dl = make_loaders(
        args.batch_size, args.data_dir, args.num_workers, device
    )
    model = build_model(variant, patch_size=args.patch_size, rope=args.rope,
                        fused=args.fused, rope_angles=args.rope_angles,
                        rope_angle_scale=args.rope_angle_scale,
                        rope_turns=args.rope_turns).to(device)
    n_params = count_params(model)
    print(f"  params: {n_params / 1e6:.2f} M ({n_params:,})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * min(args.warmup_epochs, args.epochs)
    lr_strategy = build_lr_strategy(args, optimizer, total_steps, warmup_steps)

    epochs_log: list[dict] = []
    best_acc, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_dl, optimizer, lr_strategy, device, args.grad_clip,
                             amp=args.amp)
        ev = evaluate(model, test_dl, device, amp=args.amp)
        lr_strategy.step_per_epoch(tr["loss"])
        epochs_log.append({
            "epoch": ep,
            "train_loss": tr["loss"], "train_acc": tr["acc"],
            "test_loss": ev["loss"], "test_acc": ev["acc"],
            "lr_end": tr["lr_end"], "train_wall_s": tr["wall_s"],
        })
        print(f"  ep {ep:3d}  trL {tr['loss']:.4f} trA {tr['acc'] * 100:5.2f}  "
              f"teL {ev['loss']:.4f} teA {ev['acc'] * 100:5.2f}  "
              f"lr {tr['lr_end']:.2e}  {tr['wall_s']:.1f}s")
        if ev["acc"] > best_acc:
            best_acc = ev["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    ckpt_path = args.out / f"ckpt_{variant}.pt"
    torch.save({"variant": variant, "state_dict": best_state, "best_test_acc": best_acc}, ckpt_path)
    print(f"  saved best ckpt -> {ckpt_path}  (test_acc {best_acc * 100:.2f})")

    model.load_state_dict(best_state)
    model.eval()

    # Release the optimizer and gradients before measuring. They are still resident
    # from training and land inside the peak-memory scope, so the reported figure was
    # the *training* footprint, not the inference one: AdamW's two moments plus grads
    # are 3 fp32 copies of the parameters, which at 2.71 M params inflated peak memory
    # by 31.1 MiB against a predicted 31.0 (152.5 -> 121.4 MiB on the 2-directional
    # cell). The offset scales with parameter count, so it barely moved comparisons
    # between these matched-budget variants, but every absolute number carried it.
    optimizer.zero_grad(set_to_none=True)
    del optimizer, train_dl, test_dl
    if device.type == "cuda":
        torch.cuda.empty_cache()

    n_tokens = (32 // args.patch_size) ** 2 + 1
    x = torch.randn(args.eff_batch, 3, 32, 32, device=device)
    with torch.inference_mode():
        eff = measure(lambda inp, m=model: m(inp), x, device, warmup=3, repeats=10)
    print(f"  efficiency: latency {eff['latency_ms']:.2f} ms  peak {eff['peak_mib']:.1f} MiB "
          f"(B={args.eff_batch}, T={n_tokens})")

    # Record what the rotary *actually* was, read off the built model rather than
    # from args: --rope-angles is None when the caller lets the per-operator default
    # decide, and that default is today's code, not this run's. A checkpoint trained
    # under an older default then rebuilds at the wrong width, which is exactly what
    # broke a re-measure pass over ten directories (see remeasure_efficiency's
    # build_matching). Reading it off the model cannot drift.
    rotary_used = any(getattr(m, "num_rope_angles", 0) > 0 for m in model.modules())

    return {
        "variant": variant,
        "params": n_params,
        "rope_angles_used": rotary_used,
        "best_test_acc": best_acc,
        "final_train_acc": epochs_log[-1]["train_acc"],
        "final_test_acc": epochs_log[-1]["test_acc"],
        "mean_train_wall_s": float(np.mean([r["train_wall_s"] for r in epochs_log])),
        "efficiency": eff,
        "eff_batch": args.eff_batch,
        "eff_tokens": n_tokens,
        "epochs": epochs_log,
    }


def write_results_json(out: Path, results: dict, args) -> None:
    cfg = {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": 0.05, "seed": args.seed, "device": args.device,
        "warmup_epochs": args.warmup_epochs, "grad_clip": args.grad_clip,
        "lr_schedule": args.lr_schedule, "steps_per_epoch": CIFAR10_TRAIN_N // args.batch_size,
        "patch_size": args.patch_size, "eff_batch": args.eff_batch, "rope": args.rope,
        "fused": args.fused, "amp": args.amp,
        "rope_angles": args.rope_angles,
    }
    if args.lr_schedule == "plateau":
        cfg.update({
            "plateau_factor": args.plateau_factor, "plateau_patience": args.plateau_patience,
            "plateau_threshold": args.plateau_threshold, "plateau_min_lr": args.plateau_min_lr,
            "plateau_loss_ema_alpha": args.plateau_loss_ema_alpha,
        })
    (out / "results.json").write_text(json.dumps({"config": cfg, "variants": results}, indent=2))


def load_config(path: Path) -> dict:
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for k in _PATH_ARGS:
        if k in cfg and cfg[k] is not None and not isinstance(cfg[k], Path):
            cfg[k] = Path(cfg[k])
    return cfg


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()
    config_defaults = load_config(pre_args.config) if pre_args.config else {}

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=Path, default=Path("result/cifar10_compare"))
    ap.add_argument("--variants", nargs="+", default=["vit_mamba3_vssd"], choices=list(VARIANTS))
    ap.add_argument("--device", choices=["cuda", "cpu"],
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    ap.add_argument("--patch-size", type=int, default=1,
                    help="ViT patch side (px): 4->T=65, 2->T=257, 1->T=1025.")
    ap.add_argument("--rope", action=argparse.BooleanOptionalAction, default=True,
                    help="give the mamba mixers 2-D RoPE on B/C. --no-rope reproduces "
                         "the originally published Table 1 rows, which ran without it.")
    ap.add_argument("--no-cudnn", action="store_true",
                    help="Disable cuDNN. Escape hatch for a broken cuDNN install: this "
                    "venv currently raises CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH on "
                    "any conv2d, though it ran fine on 2026-08-07 and the venv has not "
                    "changed since 2026-07-08. Only the patch-embed conv uses cuDNN here "
                    "-- the SSD/VSSD matmuls do not -- so the cost is negligible.")
    ap.add_argument("--rope-turns", type=float, default=None,
                    help="Fix the rotary's total angular spread to this many turns over "
                    "the sequence: theta_j = j*2*pi*n/T, independent of T. Replaces the "
                    "learned cumulative angle, whose spread grows with T and decoheres "
                    "the pooled VSSD state. n=1 measured best at T=1025.")
    ap.add_argument("--rope-angle-scale", type=float, default=1.0,
                    help="Scale the VSSD rotary's per-token angle increment. theta is a "
                    "plain cumsum there, so its spread grows linearly with T; 65/1025 gives "
                    "T=1025 the spread T=65 had. Diagnostic for whether spread, not the "
                    "rotary itself, is what breaks training.")
    ap.add_argument("--rope-angles", action=argparse.BooleanOptionalAction, default=None,
                    help="Mamba-3's own complex-SSM rotary inside the operator, which "
                         "stacks with --rope. Unset means the per-operator default: on "
                         "for NC-SSD, off for the scan variants (see models.py).")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                    help="bf16 autocast, used uniformly across every variant. --no-amp "
                         "trains in fp32, for numerical debugging only.")
    ap.add_argument("--fused", action=argparse.BooleanOptionalAction, default=True,
                    help="use the Triton SSD kernel for bidirectional SSD (faster, and "
                         "what published numbers used). --no-fused runs the equivalent "
                         "PyTorch path, for CPU runs or kernel debugging.")
    ap.add_argument("--eff-batch", type=int, default=128,
                    help="Batch for the latency/peak-mem probe (128 matches the existing Table 1 rows).")
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lr-schedule", choices=["cosine", "plateau"], default="plateau")
    ap.add_argument("--plateau-factor", type=float, default=0.5)
    ap.add_argument("--plateau-patience", type=int, default=5)
    ap.add_argument("--plateau-threshold", type=float, default=1e-3)
    ap.add_argument("--plateau-min-lr", type=float, default=1e-6)
    ap.add_argument("--plateau-loss-ema-alpha", type=float, default=0.3)
    if config_defaults:
        unknown = set(config_defaults) - {a.dest for a in ap._actions}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        ap.set_defaults(**config_defaults)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.no_cudnn:
        torch.backends.cudnn.enabled = False
    device = torch.device(args.device)
    if device.type == "cpu" and args.epochs > 5:
        print("[smoke] --device cpu -> shortening epochs to 2")
        args.epochs = 2

    set_seed(args.seed)
    print(f"[data] CIFAR-10: batch_size={args.batch_size}, "
          f"steps/epoch={CIFAR10_TRAIN_N // args.batch_size} "
          f"(loaders rebuilt per variant so every variant sees one shuffle order)")

    results = {v: run_variant(v, args, device) for v in args.variants}
    write_results_json(args.out, results, args)
    print(f"\n[done] artifacts -> {args.out}")
    for v, r in results.items():
        e = r["efficiency"]
        print(f"  {v}: acc {r['best_test_acc'] * 100:.2f}%  "
              f"lat {e['latency_ms']:.1f} ms  mem {e['peak_mib']:.1f} MiB")


if __name__ == "__main__":
    main()
