"""Re-measure latency / peak memory for finished runs, from their checkpoints.

``run_cifar`` measures efficiency inline, at the end of each variant, from the
best checkpoint it just saved. That is convenient but couples a timing to whatever
else happened to be on the GPU at that moment, and it cannot be redone without
retraining.

This tool decouples them. Every ``ckpt_<variant>.pt`` plus the ``config`` block of
``results.json`` is enough to rebuild the exact model and re-time it, so a timing
can be re-taken whenever the GPU is actually idle -- and it runs under
``assert_gpu_exclusive``, so it refuses rather than reporting a contended number.

Prints old against new for every cell, because agreement is the point: a cell that
moves materially means the original timing was taken under load and the table was
wrong. Nothing is written unless ``--write`` is passed.

    uv run python -m eval.remeasure_efficiency result/vm3_t1025_norope --write
    uv run python -m eval.remeasure_efficiency result/vm3_t1025_* --write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eval.efficiency import count_params, measure
from eval.models import build_model


def remeasure_dir(out: Path, device: torch.device, write: bool) -> int:
    rj = out / "results.json"
    if not rj.exists():
        print(f"{out}: no results.json (run unfinished?) -- skipped")
        return 0
    doc = json.loads(rj.read_text())
    cfg, variants = doc["config"], doc["variants"]
    n_tokens = (32 // cfg["patch_size"]) ** 2 + 1
    changed = 0

    for variant, rec in variants.items():
        ckpt = out / f"ckpt_{variant}.pt"
        if not ckpt.exists():
            print(f"  {variant:22s} no checkpoint -- skipped")
            continue
        model = build_model(variant, patch_size=cfg["patch_size"], rope=cfg["rope"],
                            fused=cfg["fused"], rope_angles=cfg["rope_angles"]).to(device)
        # weights_only=True: these checkpoints hold only tensors, a str and a float,
        # so there is no reason to let torch.load unpickle arbitrary objects.
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["state_dict"])
        model.eval()
        x = torch.randn(cfg["eff_batch"], 3, 32, 32, device=device)
        with torch.inference_mode():
            # Bind the model as a default arg rather than closing over it: the name is
            # deleted below to free VRAM between variants, and a closure would then be
            # left pointing at an unbound name.
            eff = measure(lambda inp, m=model: m(inp), x, device, warmup=3, repeats=10)

        old = rec.get("efficiency") or {}
        d_lat = eff["latency_ms"] - old.get("latency_ms", float("nan"))
        d_mem = eff["peak_mib"] - old.get("peak_mib", float("nan"))
        print(f"  {variant:22s} lat {old.get('latency_ms', float('nan')):7.2f} ->"
              f" {eff['latency_ms']:7.2f} ms ({d_lat:+.2f})   "
              f"mem {old.get('peak_mib', float('nan')):7.1f} -> {eff['peak_mib']:7.1f} MiB ({d_mem:+.1f})")
        rec["efficiency"] = eff
        rec["eff_batch"], rec["eff_tokens"] = cfg["eff_batch"], n_tokens
        rec["params"] = count_params(model)
        changed += 1
        del model, x
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if write and changed:
        rj.write_text(json.dumps(doc, indent=2))
        print(f"  wrote {rj}")
    elif changed:
        print("  (dry run -- pass --write to update results.json)")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path, help="run directories to re-measure")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--write", action="store_true",
                    help="update results.json in place (default: report only)")
    args = ap.parse_args()

    device = torch.device(args.device)
    total = 0
    for d in args.dirs:
        print(f"\n{d}")
        total += remeasure_dir(d, device, args.write)
    print(f"\nre-measured {total} cell(s)")


if __name__ == "__main__":
    main()
