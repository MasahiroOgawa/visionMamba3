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
import time
from pathlib import Path

import torch

from eval.efficiency import GpuNotExclusive, count_params, measure
from eval.models import build_model


def wait_for_exclusive_gpu(device: torch.device, timeout_s: int) -> None:
    """Block until no other process is computing on the GPU.

    assert_gpu_exclusive is a hard failure by design: a one-shot measurement should
    refuse rather than report a contended timing. But this tool usually runs at the
    tail of a chain, seconds after a training job exits, and a CUDA context takes a
    moment to tear down -- so a scripted idle-check can pass and the measurement still
    collide. That is not hypothetical: it aborted a pass over fifteen directories with
    a 10 MiB context, i.e. a process merely starting or ending, and wrote nothing.

    Waiting turns that race into a delay. The guard downstream still fires if the GPU
    never clears, so a genuinely busy machine still refuses instead of lying.
    """
    if device.type != "cuda" or timeout_s <= 0:
        return
    for waited in range(0, timeout_s, 10):
        try:
            from eval.efficiency import assert_gpu_exclusive
            assert_gpu_exclusive(device)
            return
        except GpuNotExclusive:
            if waited == 0:
                print("  GPU not exclusive yet; waiting for it to clear...")
            time.sleep(10)
    print(f"  still not exclusive after {timeout_s}s; measuring will refuse per cell")


def build_matching(variant: str, cfg: dict, state: dict):
    """Rebuild the architecture the checkpoint was actually trained with.

    ``results.json`` records the *CLI* value of ``--rope-angles``, which is ``None``
    when the caller let the per-operator default decide. That default lives in
    ``_MAMBA3_ROTARY_DEFAULT``, i.e. in today's code, and it has already changed once
    -- so replaying ``None`` through ``build_model`` can produce a different width
    than the run being re-measured: the rotary adds ``state_dim/2`` rows to each
    block's projection, and the mismatch surfaces as a load_state_dict size error.

    So try the recorded value first (nothing changes when the config is unambiguous),
    then each explicit setting, and keep whichever matches the checkpoint's own
    shapes. The checkpoint is the only authority on what was trained.
    """
    tried = []
    for ra in (cfg["rope_angles"], False, True):
        if ra in tried:
            continue
        tried.append(ra)
        model = build_model(variant, patch_size=cfg["patch_size"], rope=cfg["rope"],
                            fused=cfg["fused"], rope_angles=ra)
        have = model.state_dict()
        if all(k in have and have[k].shape == v.shape for k, v in state.items()):
            return model, ra
    raise RuntimeError(
        f"no rope_angles setting in {tried} reproduces {variant}'s checkpoint shapes; "
        "the checkpoint predates a change this tool cannot infer"
    )


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
        # weights_only=True: these checkpoints hold only tensors, a str and a float,
        # so there is no reason to let torch.load unpickle arbitrary objects.
        state = torch.load(ckpt, map_location=device, weights_only=True)["state_dict"]
        try:
            model, used = build_matching(variant, cfg, state)
        except RuntimeError as e:
            # One unrebuildable checkpoint must not cost the other cells their
            # re-measurement: a single mismatch in this loop once aborted a pass over
            # ten directories, leaving the table half-corrected and nothing written.
            print(f"  {variant:22s} SKIPPED: {e}")
            continue
        if used != cfg["rope_angles"]:
            print(f"  {variant:22s} note: rebuilt with rope_angles={used}, not the "
                  f"config's {cfg['rope_angles']!r} (see build_matching)")
        model = model.to(device)
        model.load_state_dict(state)
        model.eval()
        x = torch.randn(cfg["eff_batch"], 3, 32, 32, device=device)
        try:
            with torch.inference_mode():
                # Bind the model as a default arg rather than closing over it: the name
                # is deleted below to free VRAM between variants, and a closure would
                # then be left pointing at an unbound name.
                eff = measure(lambda inp, m=model: m(inp), x, device, warmup=3, repeats=10)
        except GpuNotExclusive as e:
            # This, not build_matching, is where contention surfaces -- measure() is the
            # only call here that touches the GPU. Skipping one cell keeps the rest of
            # the pass, and the cell simply keeps its previous number rather than
            # gaining a contended one.
            print(f"  {variant:22s} SKIPPED: {str(e).splitlines()[0]}")
            del model, x
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

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
    ap.add_argument("--wait-gpu", type=int, default=600, metavar="SEC",
                    help="Wait up to SEC for the GPU to become exclusive (0 disables).")
    ap.add_argument("--write", action="store_true",
                    help="update results.json in place (default: report only)")
    args = ap.parse_args()

    device = torch.device(args.device)
    wait_for_exclusive_gpu(device, args.wait_gpu)
    total = 0
    for d in args.dirs:
        print(f"\n{d}")
        total += remeasure_dir(d, device, args.write)
    print(f"\nre-measured {total} cell(s)")


if __name__ == "__main__":
    main()
