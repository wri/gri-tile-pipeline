"""Tile availability checking via obstore batch listing.

Instead of N+1 ``head_object`` calls, we list the relevant S3 prefix in
1-3 paginated calls, collect all existing keys into a set, then compute
the set difference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

import obstore as obs
from loguru import logger

from gri_tile_pipeline.storage.obstore_utils import from_dest
from gri_tile_pipeline.storage.tile_paths import prediction_key, raw_ard_keys


def _list_keys(store, prefix: str) -> Set[str]:
    """List all object keys under *prefix* and return as a set."""
    keys: Set[str] = set()
    result = obs.list(store, prefix=prefix)
    for batch in result:
        for meta in batch:
            keys.add(meta["path"])
    return keys


def check_availability(
    tiles: List[Dict[str, Any]],
    dest: str,
    *,
    check_type: str = "raw_ard",
    region: str = "us-west-2",
) -> Dict[str, List[Dict[str, Any]]]:
    """Check which tiles already have outputs on S3.

    Args:
        tiles: List of tile dicts (year, lon, lat, X_tile, Y_tile).
        dest: S3 destination prefix or local path.
        check_type: ``"raw_ard"`` or ``"predictions"``.
        region: AWS region for the S3 bucket.

    Returns:
        ``{"existing": [...], "missing": [...]}`` where each list
        contains the original tile dicts.
    """
    store = from_dest(dest, region=region)

    # Group tiles by year to minimize listing calls
    by_year: Dict[int, List[Dict[str, Any]]] = {}
    for t in tiles:
        by_year.setdefault(t["year"], []).append(t)

    existing: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for year, year_tiles in sorted(by_year.items()):
        if check_type == "predictions":
            prefix = f"{year}/tiles/"
        else:
            prefix = f"{year}/raw/"

        logger.debug(f"Listing keys under {prefix} for {len(year_tiles)} tiles")

        try:
            all_keys = _list_keys(store, prefix)
        except Exception as e:
            logger.warning(f"Failed to list {prefix}: {e} — marking all as missing")
            missing.extend(year_tiles)
            continue

        logger.debug(f"Found {len(all_keys)} keys under {prefix}")

        # If the listing is very large (>100K objects), fall back to
        # per-tile-directory listing for bounded queries.
        use_per_tile = len(all_keys) > 100_000

        for t in year_tiles:
            x, y = t["X_tile"], t["Y_tile"]

            if check_type == "predictions":
                expected = [prediction_key(year, x, y)]
            else:
                expected = raw_ard_keys(year, x, y)

            if use_per_tile:
                # Re-list just this tile's directory
                if check_type == "predictions":
                    tile_prefix = f"{year}/tiles/{x}/{y}/"
                else:
                    tile_prefix = f"{year}/raw/{x}/{y}/"
                try:
                    tile_keys = _list_keys(store, tile_prefix)
                except Exception:
                    missing.append(t)
                    continue
                tile_exists = all(k in tile_keys for k in expected)
            else:
                tile_exists = all(k in all_keys for k in expected)

            if tile_exists:
                existing.append(t)
            else:
                missing.append(t)

    logger.info(
        f"Availability check: {len(existing)} existing, {len(missing)} missing "
        f"out of {len(tiles)} tiles"
    )
    return {"existing": existing, "missing": missing}


def filter_missing_tiles(
    tiles: List[Dict[str, Any]],
    dest: str,
    *,
    check_type: str = "raw_ard",
    region: str = "us-west-2",
) -> List[Dict[str, Any]]:
    """Convenience wrapper: return only the tiles that are missing from *dest*."""
    result = check_availability(tiles, dest, check_type=check_type, region=region)
    return result["missing"]
