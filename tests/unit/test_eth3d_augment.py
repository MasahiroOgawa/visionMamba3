"""Phase-C augmentation must not decouple RGB from its ground-truth depth.

A horizontal flip applied to the image but not the depth map does not crash, does not
change any tensor shape, and does not show up in the training loss as anything but slightly
worse convergence -- it silently trains the model against mirrored ground truth. That is the
same failure mode as the edge-loss shape bug that voided an earlier measurement, so it is
pinned here rather than left to a code review.

The augmentation ranges themselves (crop 0.6-1.0, flip p=0.5, brightness/contrast/saturation
0.6-1.4, hue +-0.1) come from the reference implementation. Running the final stage with a
weaker augmentation than that -- or, as happened before 2026-08-13, with none at all because
the runner never passed --augment -- is a different experiment from the one whose number we
are trying to reproduce.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("torchvision", reason="colour jitter needs torchvision")
from depth.eth3d import load_scene  # noqa: E402

SCENE = Path("data/eth3d/courtyard")
pytestmark = pytest.mark.skipif(not SCENE.exists(),
                                reason=f"{SCENE} not unpacked; ETH3D is a manual download")


class _CentreCropFlipRng:
    """Deterministic stand-in: full-square centre crop, flip on, identity colour jitter.

    Returning the midpoint from ``randint`` reproduces the ``rng=None`` centre crop exactly,
    so the augmented view differs from the unaugmented one by the flip and nothing else --
    which is what makes the comparison below a clean equality rather than an approximation.
    """

    def __init__(self) -> None:
        self.n = 0

    def uniform(self, a: float, b: float) -> float:
        self.n += 1
        if self.n == 1:          # crop scale: take the whole square
            return b
        return 0.0 if a < 0 else 1.0   # hue 0, brightness/contrast/saturation 1

    def randint(self, a: int, b: int) -> int:
        return (a + b) // 2

    def random(self) -> float:
        return 0.0               # < 0.5, so flip


def test_flip_moves_depth_and_valid_with_the_image() -> None:
    plain = load_scene(SCENE, max_images=1, image_size=126, with_depth=True)
    flipped = load_scene(SCENE, max_images=1, image_size=126, with_depth=True,
                         rng=_CentreCropFlipRng(), crop_scale=(1.0, 1.0))

    # RGB goes through torchvision's adjust_* even at identity factors, so allow float slack.
    assert torch.allclose(flipped.images, torch.flip(plain.images, dims=[-1]), atol=2e-3)
    # Depth and mask are pure tensor ops, so these must match exactly.
    assert torch.equal(flipped.depth, torch.flip(plain.depth, dims=[-1]))
    assert torch.equal(flipped.valid, torch.flip(plain.valid, dims=[-1]))


def test_augmentation_actually_perturbs_and_stays_well_formed() -> None:
    import random

    plain = load_scene(SCENE, max_images=2, image_size=126, with_depth=True)
    aug = load_scene(SCENE, max_images=2, image_size=126, with_depth=True,
                     rng=random.Random(0))

    assert aug.images.shape == plain.images.shape
    assert not torch.allclose(aug.images, plain.images), "augmentation had no effect"
    assert aug.images.min() >= 0.0 and aug.images.max() <= 1.0, "jitter left [0, 1]"
    assert torch.isfinite(aug.depth).all()
    assert torch.equal(aug.valid, aug.depth > 0), "mask and depth disagree after augmenting"
