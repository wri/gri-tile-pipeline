#!/usr/bin/env python3
"""Hydrate example/golden/ with ARD + reference TIFs for the parity tests.

Pulls the three golden tiles (1000X798Y / 799Y / 800Y) from S3 into the
local layout expected by tests/parity/test_golden_parity.py and
tests/parity/test_golden_diagnostics.py. Without this data those tests
skip with "Golden test data not found".

Also pulls one extra reference tile (1000X871Y) into example/raw_v2/,
using the same set of raw ARD sources as the golden tiles above (s2_10,
s2_20, s1, dem, s2_dates, clouds) plus its FINAL tif - just rooted at
example/sample_ard/raw_v2/ instead of example/golden/, and with the tif written
flat rather than nested. Independent of the golden fixture set above.

Each tile carries its own ARD year, so tiles can be pulled from
different years in a single run.

Usage:
    AWS_PROFILE=AWSAdministratorAccess-058755926933 \\
        uv run python scripts/download_golden.py

    # Pick a different bucket:
    uv run python scripts/download_golden.py --dest s3://other-bucket

Writes (golden tiles):
    example/golden/raw/s2_10/{tile}.hkl
    example/golden/raw/s2_20/{tile}.hkl
    example/golden/raw/s1/{tile}.hkl
    example/golden/raw/misc/dem_{tile}.hkl
    example/golden/raw/misc/s2_dates_{tile}.hkl
    example/golden/raw/clouds/clouds_{tile}.hkl
    example/golden/{tile}_FINAL.tif

Writes (reference tile - same source set as golden tiles):
    example/sample_ard/raw_v2/s2_10/1000X871Y.hkl
    example/sample_ard/raw_v2/s2_20/1000X871Y.hkl
    example/sample_ard/raw_v2/s1/1000X871Y.hkl
    example/sample_ard/raw_v2/misc/dem_1000X871Y.hkl
    example/sample_ard/raw_v2/misc/s2_dates_1000X871Y.hkl
    example/sample_ard/raw_v2/clouds/clouds_1000X871Y.hkl
    example/sample_ard/raw_v2/1000X871Y_FINAL.tif

example/ is gitignored, so this only touches the local working tree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import obstore as obs
from obstore.store import ObjectStore

from gri_tile_pipeline.storage.obstore_utils import from_dest
from gri_tile_pipeline.storage.tile_paths import (
    prediction_key,
    raw_ard_keys_by_source,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
EXAMPLE_DIR: Path = REPO_ROOT / "example"
GOLDEN_DIR: Path = EXAMPLE_DIR / "golden"
GOLDEN_RAW: Path = GOLDEN_DIR / "raw"
RAW_V2_DIR: Path = EXAMPLE_DIR / "sample_ard" / "raw_v2"

# Matches tests/conftest.py:GOLDEN_TILES. Each entry is (x, y, year) - the
# year travels with the tile so tiles can come from different ARD years.
TILES: list[tuple[int, int, int]] = [
    (1000, 798, 2023),
    (1000, 799, 2023),
    (1000, 800, 2023),
]

# Extra reference tile(s): the same raw ARD source set as the golden
# tiles above, plus the FINAL tif, written into example/raw_v2/. Not part
# of the golden fixture set. Each entry is (x, y, year).
REFERENCE_TIF_TILES: list[tuple[int, int, int]] = [
    (1000, 871, 2023),
]

DEFAULT_DEST: str = "s3://wri-restoration-geodata-ttc"


def _raw_local_path_for_source(raw_root: Path, src: str, x: int, y: int) -> Path:
    """Map ARD source name → local path under a raw/ root folder.

    Mirrors the layout that tests/parity/test_golden_parity.py:load_golden_tile
    reads from. The on-disk layout for the "misc"/"clouds" sources differs
    from the S3 key structure, so we can't just mirror keys verbatim.
    """
    tag = f"{x}X{y}Y"
    if src == "dem":
        return raw_root / "misc" / f"dem_{tag}.hkl"
    if src == "s2_dates":
        return raw_root / "misc" / f"s2_dates_{tag}.hkl"
    if src == "clouds":
        return raw_root / "clouds" / f"clouds_{tag}.hkl"
    return raw_root / src / f"{tag}.hkl"


def _golden_local_path_for_source(src: str, x: int, y: int) -> Path:
    """Map ARD source name → local path under example/golden/raw/."""
    return _raw_local_path_for_source(GOLDEN_RAW, src, x, y)


def _download(store: ObjectStore, key: str, dest_path: Path, *, force: bool) -> tuple[bool, str]:
    """Download a single object. Returns (success, status)."""
    rel = dest_path.relative_to(REPO_ROOT)
    if dest_path.exists() and not force:
        return True, f"[skip exists] {rel}"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = bytes(obs.get(store, key).bytes())
    except FileNotFoundError:
        return False, f"[MISSING on S3] {key}"
    dest_path.write_bytes(data)
    return True, f"[ok] {key} -> {rel} ({len(data):,} bytes)"


def _download_golden_tile(
    store: ObjectStore,
    x: int,
    y: int,
    year: int,
    *,
    force: bool,
    missing: list[str],
) -> None:
    """Download all raw ARD sources + the FINAL tif for one golden tile."""
    for src, key in raw_ard_keys_by_source(year, x, y).items():
        local = _golden_local_path_for_source(src, x, y)
        ok, status = _download(store, key, local, force=force)
        print(status)
        if not ok:
            missing.append(key)

    tif_key = prediction_key(year, x, y)
    tif_local = GOLDEN_DIR / f"{x}X{y}Y_FINAL.tif"
    ok, status = _download(store, tif_key, tif_local, force=force)
    print(status)
    if not ok:
        missing.append(tif_key)


def _download_reference_tile(
    store: ObjectStore,
    x: int,
    y: int,
    year: int,
    *,
    force: bool,
    missing: list[str],
) -> None:
    """Download all raw ARD sources + the FINAL tif for a tile into
    example/sample_ard/raw_v2/ - the same set of source files as
    _download_golden_tile, just rooted at RAW_V2_DIR instead of
    GOLDEN_RAW/GOLDEN_DIR."""
    tag = f"{x}X{y}Y"

    for src, key in raw_ard_keys_by_source(year, x, y).items():
        local = _raw_local_path_for_source(RAW_V2_DIR, src, x, y)
        ok, status = _download(store, key, local, force=force)
        print(status)
        if not ok:
            missing.append(key)

    tif_key = prediction_key(year, x, y)
    tif_local = RAW_V2_DIR / f"{tag}_FINAL.tif"
    ok, status = _download(store, tif_key, tif_local, force=force)
    print(status)
    if not ok:
        missing.append(tif_key)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dest", default=DEFAULT_DEST,
                    help=f"S3 root URI (default: {DEFAULT_DEST})")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--profile", default=None,
                    help="AWS profile (else default credential chain).")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if local file already exists.")
    args = ap.parse_args()

    store = from_dest(args.dest, region=args.region, profile=args.profile)
    total = len(TILES) + len(REFERENCE_TIF_TILES)
    print(f"Downloading {total} object(s) from {args.dest}")
    print(f"Golden target: {GOLDEN_DIR.relative_to(REPO_ROOT)}")
    print(f"Raw v2 target: {RAW_V2_DIR.relative_to(REPO_ROOT)} (same source set as golden)")
    print()

    missing: list[str] = []

    for x, y, year in TILES:
        _download_golden_tile(store, x, y, year, force=args.force, missing=missing)

    for x, y, year in REFERENCE_TIF_TILES:
        _download_reference_tile(store, x, y, year, force=args.force, missing=missing)

    print()
    if missing:
        print(f"[FAIL] {len(missing)} object(s) were not present on S3:")
        for k in missing:
            print(f"  - {k}")
        return 1
    print(
        f"[ok] Hydrated {GOLDEN_DIR.relative_to(REPO_ROOT)} and "
        f"{RAW_V2_DIR.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())