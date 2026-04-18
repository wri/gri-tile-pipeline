"""Tile availability checking via concurrent obstore head_async calls.

For each tile, we check the existence of its expected S3 keys using
``head_async`` with an ``asyncio.Semaphore`` to bound concurrency.
This is O(N) in the number of tiles rather than O(total_objects) for
prefix listing.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import obstore as obs
from loguru import logger

from gri_tile_pipeline.storage.obstore_utils import from_dest
from gri_tile_pipeline.storage.tile_paths import prediction_key, raw_ard_keys

MAX_CONCURRENT = 50


def _expected_keys(tile: Dict[str, Any], check_type: str) -> List[str]:
    """Return the S3 keys that must exist for a tile to be considered available."""
    year, x, y = tile["year"], tile["X_tile"], tile["Y_tile"]
    if check_type == "predictions":
        return [prediction_key(year, x, y)]
    return raw_ard_keys(year, x, y)


async def _check_tiles_async(
    store,
    tiles: List[Dict[str, Any]],
    check_type: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Check tile existence via concurrent head_async calls."""
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def check_one(tile: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        keys = _expected_keys(tile, check_type)
        async with sem:
            for key in keys:
                try:
                    await obs.head_async(store, key)
                except FileNotFoundError:
                    return (tile, False)
                except Exception as e:
                    logger.debug(f"head_async error for {key}: {e}")
                    return (tile, False)
            return (tile, True)

    results = await asyncio.gather(*[check_one(t) for t in tiles])

    existing = [t for t, ok in results if ok]
    missing = [t for t, ok in results if not ok]
    return {"existing": existing, "missing": missing}


def check_availability(
    tiles: List[Dict[str, Any]],
    dest: str,
    *,
    check_type: str = "raw_ard",
    region: str = "us-west-2",
    store=None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Check which tiles already have outputs on S3.

    Uses concurrent ``head_async`` calls to check each tile's expected
    keys, bounded by a semaphore to prevent S3 throttling.

    Args:
        tiles: List of tile dicts (year, lon, lat, X_tile, Y_tile).
        dest: S3 destination prefix or local path.
        check_type: ``"raw_ard"`` or ``"predictions"``.
        region: AWS region for the S3 bucket.
        store: Pre-configured obstore Store. If provided, *dest* and
            *region* are ignored for store creation.

    Returns:
        ``{"existing": [...], "missing": [...]}`` where each list
        contains the original tile dicts.
    """
    if store is None:
        store = from_dest(dest, region=region)

    if not tiles:
        return {"existing": [], "missing": []}

    logger.debug(
        f"Checking availability of {len(tiles)} tiles "
        f"({check_type}, concurrency={MAX_CONCURRENT})"
    )

    result = asyncio.run(_check_tiles_async(store, tiles, check_type))

    logger.info(
        f"Availability check: {len(result['existing'])} existing, "
        f"{len(result['missing'])} missing out of {len(tiles)} tiles"
    )
    return result


def filter_missing_tiles(
    tiles: List[Dict[str, Any]],
    dest: str,
    *,
    check_type: str = "raw_ard",
    region: str = "us-west-2",
    store=None,
) -> List[Dict[str, Any]]:
    """Convenience wrapper: return only the tiles that are missing from *dest*."""
    result = check_availability(
        tiles, dest, check_type=check_type, region=region, store=store,
    )
    return result["missing"]
