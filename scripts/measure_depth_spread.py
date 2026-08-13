"""Score all four ETH3D rows on one evaluator and report the per-view spread.

Two questions this answers, neither of which the training runs recorded:

1. Is the gap between the operator arms larger than the scene's own per-view spread?
   ``result.json`` stores only the mean over the twelve held-out views, so a claim that one
   operator beats another cannot be checked from it. The paper previously asserted the arms
   were within the spread; that assertion has to be re-tested now the means have moved.

2. What is DA3-SMALL's baseline on the *matched* path, measured with current code?
   Three numbers can be produced here and only one belongs in the table, so this pins it down:

     0.0323  native two-stream: DA3's own 2*embed_dim features at the four export layers,
             through the same DualDPT head, same median alignment. THIS is the table's row --
             DA3 as designed, differing from the students only in the mixer and in having no
             ETH3D training.
     0.0453  duplicated aux: single-stream embed_dim features widened by cat([x, x], -1).
             Tempting because that is what an untrained DimBridge computes, but it is not a
             fair baseline -- it throws away DA3's second stream, so it measures a crippled
             DA3 rather than DA3's attention.
     0.0417  DA3's packaged multi-view inference(), which also selects a reference view and
             estimates cameras, so it does not isolate the mixer at all.

   Both alternatives are still computed below, purely so the distinction stays visible and
   nobody re-derives the wrong one.

Run:  uv run --extra depth python scripts/measure_depth_spread.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from depth.da3 import DEFAULT_DA3, dualdpt, dualdpt_depth, load_da3, teacher_features
from depth.eth3d import TEST_SCENE, load_scene
from depth.metrics import depth_metrics
from depth.run import EXPORT_LAYERS, _init_backbone, _load_into, _student, _survive_cudnn_mismatch

IMG_SIZE, MAX_IMAGES, DEV = 504, 12, "cuda"
METRICS = ("abs_rel", "rmse", "log10", "delta_1_25")
ARMS = [("bidirectional", "result/depth/bidirectional/ft1000/ckpt.pt"),
        ("vssd", "result/depth/vssd/ft1000/ckpt.pt"),
        ("vssd_bg", "result/depth/vssd_bg/ft1000/ckpt.pt")]


def summarise(name: str, pred, gt, valid) -> dict:
    m = depth_metrics(pred, gt, valid, per_image=True)
    row = {"arm": name, "n_images": m["n_images"]}
    for k in METRICS:
        v = m[f"{k}_per_image"]
        row[k] = m[k]
        # Sample standard deviation across views: the quantity that decides whether a gap
        # between two arms is real. stdev needs >= 2 points.
        row[f"{k}_sd"] = statistics.stdev(v) if len(v) > 1 else float("nan")
    print(f"  {name:14s} " + "  ".join(f"{k}={row[k]:.4f}+-{row[k + '_sd']:.4f}" for k in METRICS),
          flush=True)
    return row


def main() -> None:
    _survive_cudnn_mismatch()
    da3 = load_da3(DEFAULT_DA3, device=DEV)
    head = dualdpt(da3)
    scene = load_scene(Path("data") / "eth3d" / TEST_SCENE, max_images=MAX_IMAGES,
                       image_size=IMG_SIZE, with_depth=True)
    imgs, gt, valid = scene.images.to(DEV), scene.depth.to(DEV), scene.valid.to(DEV)
    rows = []

    # Baseline: DA3-SMALL's own attention, matched path, zero-shot. n=EXPORT_LAYERS asks for
    # the main two-stream output at those same layers, which is what DualDPT is designed to
    # receive; aux is the single-stream export the students are distilled against.
    vit = da3.model.backbone.pretrained
    with torch.no_grad():
        x = imgs if imgs.dim() == 5 else imgs.unsqueeze(0)
        outs, aux = vit.get_intermediate_layers(
            x, n=list(EXPORT_LAYERS), export_feat_layers=list(EXPORT_LAYERS),
            ref_view_strategy="first")
        pred = dualdpt_depth(head, [o[0] for o in outs], IMG_SIZE, IMG_SIZE).float()
        rows.append(summarise("DA3-SMALL", pred, gt, valid))
        # Recorded, not used: the crippled variant, so the gap to the real baseline is on file.
        pred_dup = dualdpt_depth(head, [torch.cat([f, f], dim=-1) for f in aux],
                                 IMG_SIZE, IMG_SIZE).float()
    rows.append(summarise("DA3-SMALL-dup-aux", pred_dup, gt, valid))

    for mixer, ckpt in ARMS:
        if not Path(ckpt).exists():
            print(f"  {mixer:14s} SKIPPED: {ckpt} missing", flush=True)
            continue
        student = _student(mixer, IMG_SIZE, DEV)
        _init_backbone(student, da3)
        _load_into(Path(ckpt), student, head)
        student.eval()
        with torch.no_grad():
            pred = dualdpt_depth(head, student.bridged(imgs), IMG_SIZE, IMG_SIZE).float()
        rows.append(summarise(mixer, pred, gt, valid))

    out = Path("result/depth/spread.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n  wrote {out}")

    # The question the table's prose turns on: is the best-vs-next gap bigger than the spread?
    ops = [r for r in rows if not r["arm"].startswith("DA3-SMALL")]
    if len(ops) >= 2:
        ops.sort(key=lambda r: r["abs_rel"])
        gap = ops[1]["abs_rel"] - ops[0]["abs_rel"]
        sd = max(r["abs_rel_sd"] for r in ops)
        print(f"  best {ops[0]['arm']} {ops[0]['abs_rel']:.4f} vs next {ops[1]['arm']} "
              f"{ops[1]['abs_rel']:.4f}: gap {gap:.4f}, largest per-view sd {sd:.4f} -> "
              f"{'gap exceeds the spread' if gap > sd else 'gap is within the spread'}")


if __name__ == "__main__":
    main()
