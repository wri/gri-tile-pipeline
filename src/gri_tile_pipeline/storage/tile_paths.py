"""Path builders for raw ARD and prediction outputs on S3."""

from __future__ import annotations


# Canonical source names for raw ARD. Used by per-source availability checks.
ARD_SOURCES = ("dem", "s1", "s2_10", "s2_20", "s2_dates", "clouds")


def raw_ard_keys_by_source(year: int, x_tile: int, y_tile: int) -> dict[str, str]:
    """Return expected S3 keys for each raw ARD source, keyed by source name."""
    tag = f"{x_tile}X{y_tile}Y"
    base = f"{year}/raw/{x_tile}/{y_tile}/raw"
    return {
        "dem":       f"{base}/misc/dem_{tag}.hkl",
        "s1":        f"{base}/s1/{tag}.hkl",
        "s2_10":     f"{base}/s2_10/{tag}.hkl",
        "s2_20":     f"{base}/s2_20/{tag}.hkl",
        "s2_dates":  f"{base}/misc/s2_dates_{tag}.hkl",
        "clouds":    f"{base}/clouds/clouds_{tag}.hkl",
    }


def raw_ard_keys(year: int, x_tile: int, y_tile: int) -> list[str]:
    """Return all expected S3 keys for a tile's raw ARD data (flat list)."""
    return list(raw_ard_keys_by_source(year, x_tile, y_tile).values())


def prediction_key(year: int, x_tile: int, y_tile: int) -> str:
    """Return the expected S3 key for a tile's prediction GeoTIFF."""
    tag = f"{x_tile}X{y_tile}Y"
    return f"{year}/tiles/{x_tile}/{y_tile}/{tag}_FINAL.tif"


def prediction_stac_key(year: int, x_tile: int, y_tile: int) -> str:
    """Return the S3 key for a tile's STAC Item sidecar (JSON).

    Lives next to the ``_FINAL.tif`` with the same stem so the pair is
    discoverable by listing the tile directory.
    """
    tag = f"{x_tile}X{y_tile}Y"
    return f"{year}/tiles/{x_tile}/{y_tile}/{tag}_FINAL.json"
