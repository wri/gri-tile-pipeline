"""Predict a single golden tile from inside the predict Docker container.

Invoked by scripts/predict_container_smoke.py via `docker run`. Expects the
repo's loaders/ and example/golden/ + models/ directories mounted at fixed
paths inside the container (see the host script for the mount layout).

Usage (inside container):
    python /gate_a.py --tile 1000X798Y --out /out/1000X798Y.tif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hickle as hkl

# Mounted paths inside the container.
GOLDEN_RAW = Path("/data/golden/raw")
MODELS = Path("/data/models")

sys.path.insert(0, "/function")  # so `from loaders.predict_tile import ...` works


def _load_tile_arrays(tile: str) -> dict:
    arrays = {
        "s2_10": hkl.load(str(GOLDEN_RAW / "s2_10" / f"{tile}.hkl")),
        "s2_20": hkl.load(str(GOLDEN_RAW / "s2_20" / f"{tile}.hkl")),
        "s1": hkl.load(str(GOLDEN_RAW / "s1" / f"{tile}.hkl")),
        "dem": hkl.load(str(GOLDEN_RAW / "misc" / f"dem_{tile}.hkl")),
        "clouds": hkl.load(str(GOLDEN_RAW / "clouds" / f"clouds_{tile}.hkl")),
        "s2_dates": hkl.load(str(GOLDEN_RAW / "misc" / f"s2_dates_{tile}.hkl")),
    }
    clm_path = GOLDEN_RAW / "clouds" / f"cloudmask_{tile}.hkl"
    if clm_path.is_file():
        arrays["clm"] = hkl.load(str(clm_path))
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from loaders.predict_tile import predict_tile_from_arrays, _write_geotiff_local

    arrays = _load_tile_arrays(args.tile)
    pred = predict_tile_from_arrays(model_path=str(MODELS), seed=42, **arrays)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_geotiff_local(str(args.out), pred, 0.0, 0.0)
    print(f"wrote {args.out} shape={pred.shape}")


if __name__ == "__main__":
    main()
