"""ETH3D depth: distil, fine-tune, evaluate. One entry point for all three mixers.

    uv run --extra depth python -m depth.run distill  --mixer vssd_bg --steps 20000 --out runs/vssd_bg
    uv run --extra depth python -m depth.run finetune --mixer vssd_bg --init runs/vssd_bg/ckpt.pt \
        --steps 500 --lr-mixer 1e-4 --lr-bridge 3e-4 --out runs/vssd_bg_ft500
    uv run --extra depth python -m depth.run finetune --mixer vssd_bg --init runs/vssd_bg_ft500/ckpt.pt \
        --steps 1000 --lr-mixer 1e-5 --lr-bridge 3e-5 --lr-head 1e-5 --unfreeze-head --augment \
        --out runs/vssd_bg_ft1000
    uv run --extra depth python -m depth.run eval --mixer vssd_bg --ckpt runs/vssd_bg_ft1000/ckpt.pt

Two fine-tune stages rather than one because that is the recipe that produced the reference
result: 500 steps at the higher rate with the head frozen, then 1000 more with the head
unfrozen at a low rate. Collapsing them into one longer run is not equivalent -- the second
stage starts from an already-adapted bridge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from depth.da3 import DEFAULT_DA3, dualdpt, dualdpt_depth, load_da3, teacher_features
from depth.eth3d import TEST_SCENE, TRAIN_SCENES, VAL_SCENES, iter_scenes, load_scene
from depth.metrics import depth_metrics
from depth.student import DepthStudent, flatten_views

EXPORT_LAYERS = (5, 7, 9, 11)


def _survive_cudnn_mismatch() -> None:
    """Fall back to cuDNN-free convolutions if the installed cuDNN cannot finalize.

    On a machine where a system-wide cuDNN outranks the wheel's own on the loader path,
    torch raises CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH on the first convolution -- here,
    a system 9.25 supplying `libcudnn_engines_tensor_ir` that the pip 9.20 does not ship.
    Only the DPT head convolves, and the non-cuDNN path costs nothing measurable at these
    sizes, so the run continues rather than dying at step 0.

    This is a workaround, not a fix. The fix is to remove the system cuDNN so the wheel's
    own is used; the message says so rather than letting a silent fallback hide it.
    """
    if not torch.cuda.is_available() or not torch.backends.cudnn.enabled:
        return
    try:
        torch.nn.functional.conv2d(
            torch.zeros(1, 1, 8, 8, device="cuda"), torch.zeros(1, 1, 3, 3, device="cuda")
        )
    except RuntimeError as e:
        if "CUDNN" not in str(e).upper():
            raise
        torch.backends.cudnn.enabled = False
        print("[depth] cuDNN unusable on this host (sublibrary version mismatch); "
              "running convolutions without it. Remove the system cuDNN to restore it.",
              flush=True)


def feature_loss(student_feats: list[Tensor], teacher_feats: list[Tensor]) -> Tensor:
    """Per-layer L2 + (1 - cosine), the distillation objective.

    L2 is divided by channel count so the two terms stay comparable as width changes;
    cosine is what actually carries direction, and L2 alone lets the student match
    magnitude while pointing elsewhere.

    Averaged over layers, not summed: summing four layers scales the gradient by four at
    the same learning rate, which is a 4x larger effective step than the reference recipe
    took and is not what its lr 3e-4 was tuned for.
    """
    loss = torch.zeros((), device=student_feats[0].device)
    for fs, ft in zip(flatten_views(student_feats), flatten_views(teacher_feats)):
        s, t = fs.float(), ft.float().detach()
        loss = loss + (s - t).pow(2).mean() / max(s.shape[-1], 1)
        loss = loss + (1.0 - (F.normalize(s, dim=-1) * F.normalize(t, dim=-1)).sum(-1).mean())
    return loss / max(len(student_feats), 1)


def silog_loss(pred: Tensor, gt: Tensor, valid: Tensor, lam: float = 0.85) -> Tensor:
    """Scale-invariant log RMSE, the standard monocular-depth objective.

    Scale-invariant on purpose: the head is being taught depth *shape*, and the global
    factor is recovered by median alignment at evaluation.
    """
    total = torch.zeros((), device=pred.device)
    n = 0
    for i in range(pred.shape[0]):
        v = valid[i]
        if v.sum() < 16:
            continue
        d = pred[i][v].clamp_min(1e-6).log() - gt[i][v].clamp_min(1e-6).log()
        total = total + (d.pow(2).mean() - lam * d.mean().pow(2)).clamp_min(0).sqrt()
        n += 1
    return total / max(n, 1)


def edge_aware_smoothness(depth: Tensor, image: Tensor) -> Tensor:
    """Penalise depth gradients where the image is smooth. ``depth`` (N,H,W), ``image`` (N,3,H,W).

    Ported from the reference recipe, which trained on SILog + 0.1 * this term. SILog is
    scale-invariant and per-pixel, so on its own it never asks neighbouring pixels to agree;
    this term supplies that, weighted down where the image itself has an edge.
    """
    assert image.shape[-3] == 3 and image.shape[:-3] == depth.shape[:-2], (
        f"expected depth (N,H,W) and image (N,3,H,W); got {tuple(depth.shape)} and "
        f"{tuple(image.shape)}. Broadcasting will silently accept a collapsed image here."
    )
    d = depth.unsqueeze(1)
    d = d / d.mean(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
    dx = (d[..., :, 1:] - d[..., :, :-1]).abs()
    dy = (d[..., 1:, :] - d[..., :-1, :]).abs()
    ix = (image[..., :, 1:] - image[..., :, :-1]).abs().mean(dim=-3, keepdim=True)
    iy = (image[..., 1:, :] - image[..., :-1, :]).abs().mean(dim=-3, keepdim=True)
    return (dx * torch.exp(-ix)).mean() + (dy * torch.exp(-iy)).mean()


def _init_backbone(student: DepthStudent, da3_model) -> None:
    """Seed the student's non-attention backbone from DA3. See DepthStudent.init_from_da3."""
    st = student.init_from_da3(da3_model)
    print(f"[weights] loaded {st['loaded']}/{st['total']} tensors  "
          f"(skipped_attn={st['skipped_attn']}, shape_mismatch={st['shape_mismatch']}, "
          f"missing_in_src={st['missing_in_src']})", flush=True)


def _student(mixer: str, img_size: int, device: str, chunk_size: int | None = None) -> DepthStudent:
    return DepthStudent(mixer=mixer, img_size=img_size, patch_size=14,
                        chunk_size=chunk_size, export_layers=EXPORT_LAYERS).to(device)


def _save(path: Path, student: DepthStudent, head: torch.nn.Module | None, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"student": student.state_dict(),
                "head": head.state_dict() if head is not None else None,
                "meta": meta}, path)


def _load_into(ckpt: Path, student: DepthStudent, head: torch.nn.Module | None) -> None:
    # weights_only=True: these checkpoints hold only state dicts and a plain metadata
    # dict, so there is no reason to let torch unpickle arbitrary objects from a file
    # that may not be ours.
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    student.load_state_dict(state["student"])
    if head is not None and state.get("head") is not None:
        head.load_state_dict(state["head"])


def cmd_distill(a) -> None:
    dev = a.device
    _survive_cudnn_mismatch()
    student = _student(a.mixer, a.img_size, dev, a.chunk_size)
    teacher = load_da3(a.teacher, device=dev)
    _init_backbone(student, teacher)
    params = student.trainable("mixer")
    opt = torch.optim.AdamW(params, lr=a.lr_mixer, weight_decay=0.05)
    # eta_min is a tenth of the peak, not zero. The reference implementation anneals to
    # `lr * 0.1` in both phases and its logs end at 3e-5 and 1e-6 accordingly; ours ended at
    # exactly 0, so the final steps trained at no learning rate at all.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps,
                                                       eta_min=a.lr_mixer * 0.1)
    data = iter_scenes(Path(a.data_root), TRAIN_SCENES, n_views=a.n_views,
                       image_size=a.img_size, with_depth=False, seed=a.seed)
    print(f"[distill] mixer={a.mixer} trainable={sum(p.numel() for p in params)/1e6:.2f}M "
          f"steps={a.steps} lr={a.lr_mixer}", flush=True)
    student.train()
    for step in range(a.steps):
        imgs = next(data).images.to(dev)
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            loss = feature_loss(student.features(imgs), teacher_features(teacher, imgs, EXPORT_LAYERS))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % a.log_every == 0 or step == a.steps - 1:
            print(f"[distill] step {step:6d}/{a.steps}  loss={loss.item():.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
    _save(Path(a.out) / "ckpt.pt", student, None, {"phase": "distill", "mixer": a.mixer, "steps": a.steps})
    print(f"[distill] wrote {Path(a.out) / 'ckpt.pt'}", flush=True)


def cmd_finetune(a) -> None:
    dev = a.device
    _survive_cudnn_mismatch()
    student = _student(a.mixer, a.img_size, dev, a.chunk_size)
    da3 = load_da3(a.teacher, device=dev)
    _init_backbone(student, da3)
    head = dualdpt(da3)
    if a.init:
        _load_into(Path(a.init), student, head)

    groups = [{"params": student.trainable("mixer"), "lr": a.lr_mixer},
              {"params": student.trainable("bridge"), "lr": a.lr_bridge}]
    for p in head.parameters():
        p.requires_grad_(a.unfreeze_head)
    if a.unfreeze_head:
        groups.append({"params": list(head.parameters()), "lr": a.lr_head})
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps,
                                                       eta_min=a.lr_mixer * 0.1)

    data = iter_scenes(Path(a.data_root), TRAIN_SCENES, n_views=a.n_views,
                       image_size=a.img_size, with_depth=True, augment=a.augment, seed=a.seed)
    print(f"[finetune] mixer={a.mixer} head={'trained' if a.unfreeze_head else 'frozen'} "
          f"steps={a.steps} augment={a.augment}", flush=True)
    student.train()
    for step in range(a.steps):
        sc = next(data)
        imgs, gt, valid = sc.images.to(dev), sc.depth.to(dev), sc.valid.to(dev)
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            pred = dualdpt_depth(head, student.bridged(imgs), a.img_size, a.img_size)
            l_silog = silog_loss(pred.float(), gt, valid)
            l_edge = edge_aware_smoothness(pred.float(), imgs)
            loss = l_silog + a.lambda_edge * l_edge
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % a.log_every == 0 or step == a.steps - 1:
            print(f"[finetune] step {step:6d}/{a.steps}  silog={l_silog.item():.4f}  "
                  f"edge={l_edge.item():.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  [{sc.name}]", flush=True)
        # Periodic checkpoints so a long run that starts degrading can still be scored at its
        # best step instead of only at the last one. Depth accuracy on a held-out scene is not
        # monotone in the training loss, so the final step is not reliably the best step.
        if a.ckpt_every and step and step % a.ckpt_every == 0:
            _save(Path(a.out) / f"ckpt_{step}.pt", student, head,
                  {"phase": "finetune", "mixer": a.mixer, "steps": step,
                   "unfreeze_head": a.unfreeze_head})
            print(f"[finetune] wrote {Path(a.out) / f'ckpt_{step}.pt'}", flush=True)
    _save(Path(a.out) / "ckpt.pt", student, head,
          {"phase": "finetune", "mixer": a.mixer, "steps": a.steps, "unfreeze_head": a.unfreeze_head})
    print(f"[finetune] wrote {Path(a.out) / 'ckpt.pt'}", flush=True)


@torch.no_grad()
def cmd_eval(a) -> None:
    dev = a.device
    _survive_cudnn_mismatch()
    student = _student(a.mixer, a.img_size, dev, a.chunk_size)
    da3 = load_da3(a.teacher, device=dev)
    _init_backbone(student, da3)
    head = dualdpt(da3)
    if a.ckpt:
        _load_into(Path(a.ckpt), student, head)
    student.eval()

    scene = load_scene(Path(a.data_root) / "eth3d" / a.scene, max_images=a.max_images,
                       image_size=a.img_size, with_depth=True)
    imgs, gt, valid = scene.images.to(dev), scene.depth.to(dev), scene.valid.to(dev)
    pred = dualdpt_depth(head, student.bridged(imgs), a.img_size, a.img_size).float()
    m = depth_metrics(pred, gt, valid)
    m.update(mixer=a.mixer, scene=a.scene, ckpt=str(a.ckpt or ""))
    print(f"  {a.mixer:14s} abs_rel={m['abs_rel']:.4f}  delta<1.25={m['delta_1_25']:.4f}  "
          f"rmse={m['rmse']:.4f}  log10={m['log10']:.4f}  (n={m['n_images']})", flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(m, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--mixer", choices=["bidirectional", "vssd", "vssd_bg"], default="vssd_bg")
        p.add_argument("--data-root", type=Path, default=Path("data"))
        p.add_argument("--img-size", type=int, default=504)
        p.add_argument("--n-views", type=int, default=4)
        p.add_argument("--teacher", default=DEFAULT_DA3)
        p.add_argument("--device", default="cuda")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--log-every", type=int, default=100)
        p.add_argument("--chunk-size", type=int, default=None,
                       help="Build the scan mask in row chunks instead of a full T x T. "
                            "Cuts memory and COSTS speed (measured 3x slower at 504px), so "
                            "leave off unless a view count does not otherwise fit. Ignored "
                            "by the collapse variants, which have no T x T mask.")

    d = sub.add_parser("distill", help="Phase B: match the teacher's features")
    common(d); d.add_argument("--steps", type=int, default=20000)
    d.add_argument("--lr-mixer", type=float, default=3e-4)
    d.add_argument("--out", type=Path, required=True)
    d.set_defaults(fn=cmd_distill)

    f = sub.add_parser("finetune", help="Phase C: supervise depth against ground truth")
    common(f); f.add_argument("--steps", type=int, default=1000)
    f.add_argument("--init", type=Path, default=None)
    f.add_argument("--lr-mixer", type=float, default=1e-5)
    f.add_argument("--lr-bridge", type=float, default=3e-5)
    f.add_argument("--lr-head", type=float, default=1e-5)
    f.add_argument("--unfreeze-head", action="store_true")
    f.add_argument("--augment", action="store_true")
    f.add_argument("--ckpt-every", type=int, default=0,
                   help="Also save ckpt_<step>.pt every N steps (0 = final checkpoint only).")
    f.add_argument("--lambda-edge", type=float, default=0.1,
                   help="Weight on edge-aware smoothness; 0.1 is the reference value, 0 disables.")
    f.add_argument("--out", type=Path, required=True)
    f.set_defaults(fn=cmd_finetune)

    e = sub.add_parser("eval", help=f"score on the held-out {TEST_SCENE} scene")
    common(e); e.add_argument("--ckpt", type=Path, default=None)
    e.add_argument("--scene", default=TEST_SCENE)
    e.add_argument("--max-images", type=int, default=12)
    e.add_argument("--out", type=Path, default=None)
    e.set_defaults(fn=cmd_eval)

    a = ap.parse_args()
    if a.cmd in ("distill", "finetune") and TEST_SCENE in TRAIN_SCENES + VAL_SCENES:
        raise SystemExit(f"{TEST_SCENE} must not be in the training or validation split")
    a.fn(a)


if __name__ == "__main__":
    main()
