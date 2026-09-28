#!/usr/bin/env python3
"""Download and unpack the evaluation datasets into data/, where the experiments look for them.

  uv run python scripts/download_data.py cifar10                  # CIFAR-10 (~170 MB)
  uv run --extra depth python scripts/download_data.py eth3d      # ETH3D (~10.8 GB, ~40 GB unpacked)
  uv run --extra depth python scripts/download_data.py            # both (same as `all`)

Anything already on disk is skipped, so an interrupted run can simply be restarted. Archives
are deleted once unpacked. ETH3D is 7z, which needs the `7z` binary (p7zip).

The ETH3D scene list is read from depth.eth3d, the same split the depth experiment trains and
tests on, so the two cannot disagree about which scenes exist.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ETH3D_URL = "https://www.eth3d.net/data/{scene}_dslr_{kind}.7z"
# Archive kind -> the subdirectory it contributes to <scene>/. undistorted = the JPGs and their
# calibration; depth = ground truth at the sensor resolution.
ETH3D_KINDS = {"undistorted": "images", "depth": "ground_truth_depth"}


def fetch(url: str, dest: Path) -> Path:
    """Download to `<dest>.part` and rename on success, so a killed run never leaves a
    truncated file that looks complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"[download] {url}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 2**20:8.0f} / {total / 2**20:.0f} MB", end="", flush=True)
    print()
    part.rename(dest)
    return dest


def cifar10() -> None:
    root = DATA / "cifar10"
    if (root / "cifar-10-batches-py").exists():
        print(f"[cifar10] already present at {root}")
        return
    archive = fetch(CIFAR_URL, root / "cifar-10-python.tar.gz")
    with tarfile.open(archive) as t:
        t.extractall(root, filter="data")
    archive.unlink()
    print(f"[cifar10] ready at {root}")


def _has(scene_dir: Path, sub: str) -> bool:
    # Either layout the loader accepts: <scene>/<sub> or <scene>/<scene>/<sub>.
    return (scene_dir / sub).exists() or (scene_dir / scene_dir.name / sub).exists()


def eth3d() -> None:
    # Imported here: depth.eth3d needs the `depth` extra (Pillow), which CIFAR-10 does not.
    from depth.eth3d import TEST_SCENE, TRAIN_SCENES, VAL_SCENES

    root = DATA / "eth3d"
    # Checked per archive, not per scene: a scene whose images unpacked but whose depth did not
    # must still fetch its depth on the next run.
    todo = [(scene, kind) for scene in (*TRAIN_SCENES, *VAL_SCENES, TEST_SCENE)
            for kind, sub in ETH3D_KINDS.items() if not _has(root / scene, sub)]
    if not todo:
        print(f"[eth3d] all scenes already present at {root}")
        return
    if shutil.which("7z") is None:
        sys.exit("[eth3d] the `7z` binary is required. Install it with:\n"
                 "  sudo apt install p7zip-full")
    for scene, kind in todo:
        archive = fetch(ETH3D_URL.format(scene=scene, kind=kind), root / f"{scene}_dslr_{kind}.7z")
        # Unpack into a staging directory and move the result in only once 7z has finished, so
        # a killed extraction never leaves a half-written directory that _has() takes as done.
        stage = root / f".{scene}_{kind}.partial"
        shutil.rmtree(stage, ignore_errors=True)
        subprocess.run(["7z", "x", "-y", "-bso0", f"-o{stage}", str(archive)], check=True)
        (root / scene).mkdir(parents=True, exist_ok=True)
        # Each archive holds a top-level <scene>/ directory.
        for child in (stage / scene).iterdir():
            # A leftover from a run killed between these moves is replaced, not merged.
            shutil.rmtree(root / scene / child.name, ignore_errors=True)
            child.rename(root / scene / child.name)
        shutil.rmtree(stage)
        archive.unlink()
        print(f"[eth3d] {scene} {kind} ready")
    print(f"[eth3d] ready at {root}")


DATASETS = {"cifar10": cifar10, "eth3d": eth3d}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", choices=[*DATASETS, "all"], default="all")
    name = ap.parse_args().dataset
    for fn in DATASETS.values() if name == "all" else [DATASETS[name]]:
        fn()


if __name__ == "__main__":
    main()
