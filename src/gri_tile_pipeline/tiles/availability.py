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
from gri_tile_pipeline.storage.tile_paths import (
    ARD_SOURCES,
    prediction_key,
    raw_ard_keys,
    raw_ard_keys_by_source,
)

MAX_CONCURRENT = 50

# Sources the per-source API recognizes. "prediction" is an alias for the
# single prediction GeoTIFF.
AVAILABLE_SOURCES = tuple(ARD_SOURCES) + ("prediction",)


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
    region: str = "us-east-1",
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
    region: str = "us-east-1",
    store=None,
) -> List[Dict[str, Any]]:
    """Convenience wrapper: return only the tiles that are missing from *dest*."""
    result = check_availability(
        tiles, dest, check_type=check_type, region=region, store=store,
    )
    return result["missing"]


def _tile_id(tile: Dict[str, Any]) -> tuple[int, int, int]:
    return (int(tile["year"]), int(tile["X_tile"]), int(tile["Y_tile"]))


def _keys_for_sources(tile: Dict[str, Any], sources: tuple[str, ...]) -> dict[str, str]:
    year, x, y = tile["year"], tile["X_tile"], tile["Y_tile"]
    ard = raw_ard_keys_by_source(year, x, y)
    keys: dict[str, str] = {}
    for src in sources:
        if src == "prediction":
            keys[src] = prediction_key(year, x, y)
        elif src in ard:
            keys[src] = ard[src]
        else:
            raise ValueError(
                f"Unknown source '{src}'. Valid: {sorted(AVAILABLE_SOURCES)}"
            )
    return keys


async def _check_sources_async(
    store,
    tiles: List[Dict[str, Any]],
    sources: tuple[str, ...],
) -> Dict[tuple[int, int, int], Dict[str, bool]]:
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def check_key(key: str) -> bool:
        async with sem:
            try:
                await obs.head_async(store, key)
                return True
            except FileNotFoundError:
                return False
            except Exception as exc:
                logger.debug(f"head_async error for {key}: {exc}")
                return False

    out: Dict[tuple[int, int, int], Dict[str, bool]] = {}
    tasks = []
    index: list[tuple[tuple[int, int, int], str]] = []
    for tile in tiles:
        tid = _tile_id(tile)
        out[tid] = {}
        for src, key in _keys_for_sources(tile, sources).items():
            index.append((tid, src))
            tasks.append(check_key(key))

    results = await asyncio.gather(*tasks)
    for (tid, src), present in zip(index, results):
        out[tid][src] = present
    return out


def format_tile_report(
    tile: Dict[str, Any],
    presence: Dict[str, bool],
    dest: str,
) -> str:
    """Render a single-tile per-source availability report as human-readable text.

    ``presence`` maps source name (``dem``, ``s1``, ..., ``prediction``) to bool.
    """
    year, x, y = tile["year"], tile["X_tile"], tile["Y_tile"]
    tag = f"{x}X{y}Y"

    ard_sources = [s for s in ARD_SOURCES if s in presence]
    has_prediction = "prediction" in presence
    width = max((len(s) for s in presence), default=0)

    lines: list[str] = []
    lines.append(f"Tile {tag}  |  year {year}  |  {dest}")
    lines.append("-" * max(48, len(lines[0])))

    ard_keys = raw_ard_keys_by_source(int(year), int(x), int(y))
    for src in ard_sources:
        mark = "present" if presence[src] else "MISSING"
        lines.append(f"  {src:<{width}}  {mark:<8}  {ard_keys[src]}")

    if has_prediction:
        pkey = prediction_key(int(year), int(x), int(y))
        mark = "present" if presence["prediction"] else "MISSING"
        lines.append(f"  {'prediction':<{width}}  {mark:<8}  {pkey}")

    lines.append("-" * max(48, len(lines[0])))
    ard_present = sum(1 for s in ard_sources if presence[s])
    lines.append(f"Raw ARD:    {ard_present}/{len(ard_sources)} sources present")
    if has_prediction:
        lines.append(
            "Prediction: present" if presence["prediction"] else "Prediction: not generated"
        )
    return "\n".join(lines)


def check_availability_by_source(
    tiles: List[Dict[str, Any]],
    dest: str,
    *,
    sources: tuple[str, ...] = AVAILABLE_SOURCES,
    region: str = "us-east-1",
    store=None,
) -> Dict[tuple[int, int, int], Dict[str, bool]]:
    """Per-source availability map: ``{(year, X_tile, Y_tile): {source: present, ...}}``.

    Unlike :func:`check_availability`, which returns a binary existing/missing
    split, this tells you exactly which of (``dem``, ``s1``, ``s2_10``, ``s2_20``,
    ``s2_dates``, ``clouds``, ``prediction``) are present for each tile — so
    downstream steps can skip tiles missing essential sources while still running
    tiles whose only gap is, say, the ``clouds`` mask.
    """
    for src in sources:
        if src not in AVAILABLE_SOURCES:
            raise ValueError(
                f"Unknown source '{src}'. Valid: {sorted(AVAILABLE_SOURCES)}"
            )
    if store is None:
        store = from_dest(dest, region=region)
    if not tiles:
        return {}

    logger.debug(
        f"Per-source availability for {len(tiles)} tiles x {len(sources)} sources"
    )
    return asyncio.run(_check_sources_async(store, tiles, tuple(sources)))
