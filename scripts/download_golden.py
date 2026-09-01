#!/usr/bin/env python3
"""Hydrate example/golden/ with ARD + reference TIFs for the parity tests.

Pulls the three golden tiles (1000X798Y / 799Y / 800Y) from S3 into the
local layout expected by tests/parity/test_golden_parity.py and
tests/parity/test_golden_diagnostics.py. Without this data those tests
skip with "Golden test data not found".

Usage:
    AWS_PROFILE=AWSAdministratorAccess-058755926933 \\
        uv run python scripts/download_golden.py

    # Pick a different year or bucket:
    uv run python scripts/download_golden.py --year 2024 --dest s3://other-bucket

Writes:
    example/golden/raw/s2_10/{tile}.hkl
    example/golden/raw/s2_20/{tile}.hkl
    example/golden/raw/s1/{tile}.hkl
    example/golden/raw/misc/dem_{tile}.hkl
    example/golden/raw/misc/s2_dates_{tile}.hkl
    example/golden/raw/clouds/clouds_{tile}.hkl
    example/golden/{tile}_FINAL.tif

example/ is gitignored, so this only touches the local working tree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import obstore as obs

from gri_tile_pipeline.storage.obstore_utils import from_dest
from gri_tile_pipeline.storage.tile_paths import (
    prediction_key,
    raw_ard_keys_by_source,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
GOLDEN_DIR: Path = REPO_ROOT / "example" / "golden"
GOLDEN_RAW: Path = GOLDEN_DIR / "raw"

# Matches tests/conftest.py:GOLDEN_TILES.
TILES: list[tuple[int, int]] = [
    (1000, 798),
    (1000, 799),
    (1000, 800),
]
DEFAULT_YEAR: int = 2023
DEFAULT_DEST: str = "s3://wri-restoration-geodata-ttc"


def _local_path_for_source(src: str, x: int, y: int) -> Path:
    """Map ARD source name → local path under example/golden/raw/.

    Mirrors the layout that tests/parity/test_golden_parity.py:load_golden_tile
    reads from. The on-disk layout for the three "misc"/"clouds" sources
    differs from the S3 key structure, so we can't just mirror keys
    verbatim.
    """
    tag = f"{x}X{y}Y"
    if src == "dem":
        return GOLDEN_RAW / "misc" / f"dem_{tag}.hkl"
    if src == "s2_dates":
        return GOLDEN_RAW / "misc" / f"s2_dates_{tag}.hkl"
    if src == "clouds":
        return GOLDEN_RAW / "clouds" / f"clouds_{tag}.hkl"
    return GOLDEN_RAW / src / f"{tag}.hkl"


def _download(store: obs.Store, key: str, dest_path: Path, *, force: bool) -> tuple[bool, str]:
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dest", default=DEFAULT_DEST,
                    help=f"S3 root URI (default: {DEFAULT_DEST})")
    ap.add_argument("--year", type=int, default=DEFAULT_YEAR,
                    help=f"ARD year (default: {DEFAULT_YEAR})")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--profile", default=None,
                    help="AWS profile (else default credential chain).")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if local file already exists.")
    args = ap.parse_args()

    store = from_dest(args.dest, region=args.region, profile=args.profile)
    print(f"Downloading {len(TILES)} golden tiles for year={args.year} from {args.dest}")
    print(f"Target: {GOLDEN_DIR.relative_to(REPO_ROOT)}")
    print()

    missing: list[str] = []
    for x, y in TILES:
        for src, key in raw_ard_keys_by_source(args.year, x, y).items():
            local = _local_path_for_source(src, x, y)
            ok, status = _download(store, key, local, force=args.force)
            print(status)
            if not ok:
                missing.append(key)

        tif_key = prediction_key(args.year, x, y)
        tif_local = GOLDEN_DIR / f"{x}X{y}Y_FINAL.tif"
        ok, status = _download(store, tif_key, tif_local, force=args.force)
        print(status)
        if not ok:
            missing.append(tif_key)

    print()
    if missing:
        print(f"[FAIL] {len(missing)} object(s) were not present on S3:")
        for k in missing:
            print(f"  - {k}")
        return 1
    print(f"[ok] Hydrated {GOLDEN_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
