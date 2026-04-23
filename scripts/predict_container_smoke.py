#!/usr/bin/env python3
"""Gate A: build the predict Docker image locally and verify it on a golden tile.

Runs before any ECR push. Catches dependency-resolution, TF-import, and
graph-loading bugs in ~5 min without touching AWS.

Usage:
    python scripts/predict_container_smoke.py                 # default tile 1000X798Y
    python scripts/predict_container_smoke.py --tile 1000X799Y
    python scripts/predict_container_smoke.py --skip-build    # reuse existing image

Exits 0 on pass; non-zero with a diff summary on fail.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_TAG = "ttc-predict-local"
IN_CONTAINER_SCRIPT = REPO_ROOT / "scripts" / "_gate_a_predict_inside_container.py"
DOCKERFILE = REPO_ROOT / "docker" / "PredictDockerfile"
GOLDEN_DIR = REPO_ROOT / "example" / "golden"
MODELS_DIR = REPO_ROOT / "models"
LOADERS_DIR = REPO_ROOT / "loaders"

# Parity thresholds (same as tests/parity/test_golden_parity.py baseline).
# Gate A uses the baseline tier — a green baseline means the container's
# inference is wired correctly, regardless of whether the improved-tier
# parity tier currently passes.
PCT_WITHIN_10_MIN = 75.0
PCT_WITHIN_1_MIN = 40.0


def _build_image() -> None:
    print(f"[gate-a] docker build -t {IMAGE_TAG} -f {DOCKERFILE.relative_to(REPO_ROOT)} .")
    result = subprocess.run(
        ["docker", "build", "-t", IMAGE_TAG, "-f", str(DOCKERFILE), str(REPO_ROOT)],
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        sys.exit(f"[gate-a] docker build failed (exit {result.returncode})")


def _run_predict_in_container(tile: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / f"{tile}.tif"
    cmd = [
        "docker", "run", "--rm",
        "--entrypoint", "python",
        "-v", f"{LOADERS_DIR}:/function/loaders:ro",
        "-v", f"{MODELS_DIR}:/data/models:ro",
        "-v", f"{GOLDEN_DIR}:/data/golden:ro",
        "-v", f"{out_dir}:/out",
        "-v", f"{IN_CONTAINER_SCRIPT}:/gate_a.py:ro",
        IMAGE_TAG,
        "/gate_a.py", "--tile", tile, "--out", f"/out/{tile}.tif",
    ]
    print(f"[gate-a] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[gate-a] container predict failed (exit {result.returncode})")
    if not out_tif.is_file():
        sys.exit(f"[gate-a] expected {out_tif} but it was not written")
    return out_tif


def _compare(pred_tif: Path, ref_tif: Path) -> dict:
    from tests.parity.metrics import compare_predictions

    with rasterio.open(str(pred_tif)) as src:
        pred = src.read(1)
    with rasterio.open(str(ref_tif)) as src:
        ref = src.read(1)
    h = min(pred.shape[0], ref.shape[0])
    w = min(pred.shape[1], ref.shape[1])
    return compare_predictions(pred[:h, :w], ref[:h, :w])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile", default="1000X798Y")
    parser.add_argument("--skip-build", action="store_true", help="Reuse the existing local image")
    args = parser.parse_args()

    ref_tif = GOLDEN_DIR / f"{args.tile}_FINAL.tif"
    if not ref_tif.is_file():
        sys.exit(f"[gate-a] reference TIF not found: {ref_tif}")
    if not (MODELS_DIR / "predict_graph-172.pb").is_file():
        sys.exit(f"[gate-a] model not found under {MODELS_DIR}")

    if not args.skip_build:
        _build_image()

    with tempfile.TemporaryDirectory(prefix="gate-a-") as tmp:
        pred_tif = _run_predict_in_container(args.tile, Path(tmp))
        stats = _compare(pred_tif, ref_tif)

    print()
    print(f"[gate-a] parity vs {ref_tif.name}:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")
    print()

    failures: list[str] = []
    if stats["pct_within_10"] < PCT_WITHIN_10_MIN:
        failures.append(f"pct_within_10 = {stats['pct_within_10']:.1f}% < {PCT_WITHIN_10_MIN}%")
    if stats["pct_within_1"] < PCT_WITHIN_1_MIN:
        failures.append(f"pct_within_1 = {stats['pct_within_1']:.1f}% < {PCT_WITHIN_1_MIN}%")

    if failures:
        print("[gate-a] FAILED:")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)

    print(f"[gate-a] PASSED — container inference matches reference for {args.tile}")


if __name__ == "__main__":
    main()
