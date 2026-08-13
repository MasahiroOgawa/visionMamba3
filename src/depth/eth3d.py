"""ETH3D loading for the depth experiment: images, ground-truth depth, and the scene split.

Scope is deliberately narrow. This loads scenes that are already on disk under
``--data-root`` and does not download or unpack anything; ETH3D ships per-scene 7z archives
from https://www.eth3d.net/datasets and unpacking them is a one-off the harness should not
own. :func:`missing_scenes` reports what is absent so a run fails with a list of names
rather than a stack trace.

The one genuinely subtle part is depth alignment, and it is the reason this file is not
twenty lines. ETH3D stores GT depth as raw float32 at the *pre-undistortion* sensor
resolution, while the saved JPG is undistorted and therefore a slightly different size.
The float count does not match ``H*W`` of the JPG, so the depth shape has to be recovered
by solving for the integer pair that matches both the file length and the JPG's aspect
ratio (:func:`_infer_depth_shape`). Getting this wrong does not crash -- it silently
transposes or shears the depth map against the image, which is why it is done explicitly
and not by assuming the JPG's dimensions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor

# The split is by scene and fixed. `terrains` is the reported test set and never trains --
# the assertion in _assert_not_test enforces that, and it is the only hold-out this experiment
# needs.
#
# The other ten scenes all train. Until 2026-08-13 `relief_2` and `electro` were withheld as
# a validation split, on a comment claiming they "drive early stopping / LR schedules". They
# did not: VAL_SCENES was referenced only by its own definition and by the assertion above,
# and no code ever evaluated on it. So a fifth of the available training data was being
# discarded for a purpose nothing exercised, against a reference implementation that trains on
# all ten. VAL_SCENES stays defined, and empty, so the assertion keeps working and so the next
# person sees why it is empty rather than re-adding a split with no consumer. Re-introducing a
# validation split is fine, but it needs code that actually reads it, and it costs data.
TEST_SCENE = "terrains"
VAL_SCENES: tuple[str, ...] = ()
TRAIN_SCENES: tuple[str, ...] = (
    "courtyard", "delivery_area", "facade", "kicker",
    "office", "pipes", "playground", "relief",
    "electro", "relief_2",
)


@dataclass
class Scene:
    name: str
    images: torch.Tensor                     # (N, 3, S, S) float32 in [0, 1]
    depth: Optional[torch.Tensor] = None     # (N, S, S) float32, metres
    valid: Optional[torch.Tensor] = None     # (N, S, S) bool
    paths: list[Path] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.images.shape[0])


def _assert_not_test(scenes: Sequence[str]) -> None:
    if TEST_SCENE in tuple(scenes):
        raise ValueError(
            f"{TEST_SCENE!r} is the held-out test scene and must never be trained or "
            "tuned on. Remove it from the scene list."
        )


def missing_scenes(data_root: Path, scenes: Sequence[str]) -> list[str]:
    return [s for s in scenes if not (Path(data_root) / "eth3d" / s).exists()]


def _images_root(scene_dir: Path) -> Path:
    for c in (scene_dir / "images", scene_dir / scene_dir.name / "images", scene_dir):
        if c.exists() and any(c.rglob("*.JPG")):
            return c
    raise FileNotFoundError(f"no .JPG under {scene_dir}")


def _depth_root(scene_dir: Path) -> Optional[Path]:
    for c in (scene_dir / "ground_truth_depth" / "dslr_images",
              scene_dir / scene_dir.name / "ground_truth_depth" / "dslr_images"):
        if c.exists():
            return c
    return None


def _infer_depth_shape(n_floats: int, jpg_hw: tuple[int, int]) -> Optional[tuple[int, int]]:
    """Recover (H, W) of a raw depth file from its length and the JPG's aspect ratio.

    Depth is stored at the sensor's pre-undistortion resolution, so ``n_floats`` usually
    does not equal the JPG's ``H*W``. Aspect ratio is preserved, so solve for it.
    """
    jpg_h, jpg_w = jpg_hw
    if n_floats == jpg_h * jpg_w:
        return jpg_hw
    ratio = jpg_w / jpg_h
    h0 = int(round((n_floats / ratio) ** 0.5))
    for dh in range(-4, 5):
        h = h0 + dh
        if h <= 0 or n_floats % h:
            continue
        w = n_floats // h
        if abs(w / h - ratio) < 0.02:
            return h, w
    return None


def _crop_resize(arr: np.ndarray, box: tuple[int, int, int, int], size: int) -> np.ndarray:
    left, top, right, bottom = box
    cropped = arr[top:bottom, left:right]
    if cropped.ndim == 3:
        return np.asarray(Image.fromarray(cropped).resize((size, size), Image.BILINEAR))
    # NEAREST for depth: bilinear would interpolate across depth discontinuities and
    # invent surfaces that are not there.
    return np.asarray(Image.fromarray(cropped, mode="F").resize((size, size), Image.NEAREST),
                      dtype=np.float32)


def _square_box(h: int, w: int, rng: Optional[random.Random], scale: tuple[float, float]) -> tuple[int, int, int, int]:
    """Centre square crop, or a random one when ``rng`` is given (Phase-C augmentation)."""
    s_max = min(h, w)
    if rng is None:
        left, top = (w - s_max) // 2, (h - s_max) // 2
        return left, top, left + s_max, top + s_max
    s = min(s_max, max(32, int(round(s_max * rng.uniform(*scale)))))
    left, top = rng.randint(0, w - s), rng.randint(0, h - s)
    return left, top, left + s, top + s


def _colour_jitter(img: Tensor, rng: random.Random) -> Tensor:
    """Brightness / contrast / saturation / hue jitter on a (3, H, W) tensor in [0, 1].

    Ranges are the reference implementation's; see :func:`load_scene` for why they matter.
    RGB only -- depth must not be photometrically altered.
    """
    from torchvision.transforms.functional import (adjust_brightness, adjust_contrast,
                                                   adjust_hue, adjust_saturation)
    img = adjust_brightness(img, rng.uniform(0.6, 1.4))
    img = adjust_contrast(img, rng.uniform(0.6, 1.4))
    img = adjust_saturation(img, rng.uniform(0.6, 1.4))
    img = adjust_hue(img, rng.uniform(-0.1, 0.1))
    return img.clamp(0.0, 1.0)


def load_scene(scene_dir: Path, *, max_images: int = 4, image_size: int = 504,
               with_depth: bool = True, rng: Optional[random.Random] = None,
               crop_scale: tuple[float, float] = (0.6, 1.0)) -> Scene:
    """Load up to ``max_images`` views, square-cropped and resized to ``image_size``.

    RGB and depth are cropped with the *same* box so they stay pixel-aligned.

    When ``rng`` is given (Phase-C augmentation) the view additionally gets a horizontal
    flip with probability one half and photometric jitter. All three -- crop range, flip
    and jitter -- are the reference implementation's, which is the point: this stage is
    what took its ETH3D result to abs_rel 0.0531, and a weaker augmentation is a different
    experiment. Ours previously cropped only, over the narrower range (0.7, 1.0), with no
    flip and no jitter, which is one reason our runs sat well behind that number.

    The flip applies to depth and its validity mask as well as to RGB; the jitter applies
    to RGB alone. Flipping the image without the depth would train the model against
    mirrored ground truth.
    """
    scene_dir = Path(scene_dir)
    img_root, dep_root = _images_root(scene_dir), _depth_root(scene_dir) if with_depth else None
    paths = sorted(img_root.rglob("*.JPG"))[:max_images]
    if not paths:
        raise FileNotFoundError(f"no images in {scene_dir}")

    imgs, deps, valids = [], [], []
    for p in paths:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        box = _square_box(rgb.shape[0], rgb.shape[1], rng, crop_scale)
        # One flip decision per view, drawn before the depth branch so RGB and depth cannot
        # disagree about it.
        flip = rng.random() < 0.5 if rng is not None else False
        img = _crop_resize(rgb, box, image_size).astype(np.float32) / 255.0
        if flip:
            img = img[:, ::-1]
        view = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        imgs.append(_colour_jitter(view, rng) if rng is not None else view)

        if dep_root is None:
            continue
        dfile = dep_root / p.name
        if not dfile.exists():
            deps.append(np.zeros((image_size, image_size), np.float32))
            valids.append(np.zeros((image_size, image_size), bool))
            continue
        raw = np.fromfile(dfile, dtype=np.float32)
        shape = _infer_depth_shape(raw.size, rgb.shape[:2])
        if shape is None:
            raise ValueError(
                f"cannot infer depth shape for {dfile}: {raw.size} floats against a "
                f"{rgb.shape[1]}x{rgb.shape[0]} image. Depth and image would be misaligned."
            )
        d = raw.reshape(shape)
        # Depth is at sensor resolution; scale the box into that frame before cropping.
        sy, sx = shape[0] / rgb.shape[0], shape[1] / rgb.shape[1]
        dbox = (int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy))
        d = _crop_resize(d, dbox, image_size)
        if flip:
            d = np.ascontiguousarray(d[:, ::-1])
        valids.append(np.isfinite(d) & (d > 0))
        deps.append(np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0))

    return Scene(
        name=scene_dir.name,
        images=torch.stack(imgs).contiguous(),
        depth=torch.from_numpy(np.stack(deps)) if deps else None,
        valid=torch.from_numpy(np.stack(valids)) if valids else None,
        paths=paths,
    )


def iter_scenes(data_root: Path, scenes: Sequence[str], *, n_views: int = 4,
                image_size: int = 504, with_depth: bool = True, augment: bool = False,
                seed: int = 0) -> Iterator[Scene]:
    """Cycle scenes forever, one Scene per step. Caller decides when to stop.

    ``augment`` turns on the random square crop (ETH3D's images are far larger than
    ``image_size``, so this is a genuine crop, not an upsample).
    """
    _assert_not_test(scenes)
    missing = missing_scenes(data_root, scenes)
    if missing:
        raise FileNotFoundError(
            f"missing scenes under {Path(data_root) / 'eth3d'}: {', '.join(missing)}. "
            "Download them from https://www.eth3d.net/datasets and unpack in place."
        )
    rng = random.Random(seed)
    order = list(scenes)
    while True:
        rng.shuffle(order)
        for name in order:
            yield load_scene(
                Path(data_root) / "eth3d" / name,
                max_images=n_views, image_size=image_size, with_depth=with_depth,
                rng=rng if augment else None,
            )
