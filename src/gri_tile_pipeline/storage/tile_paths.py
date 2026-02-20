"""Path builders for raw ARD and prediction outputs on S3."""

from __future__ import annotations


def raw_ard_keys(year: int, x_tile: int, y_tile: int) -> list[str]:
    """Return all expected S3 keys for a tile's raw ARD data."""
    tag = f"{x_tile}X{y_tile}Y"
    base = f"{year}/raw/{x_tile}/{y_tile}/raw"
    return [
        f"{base}/misc/dem_{tag}.hkl",
        f"{base}/s1/{tag}.hkl",
        f"{base}/s2_10/{tag}.hkl",
        f"{base}/s2_20/{tag}.hkl",
        f"{base}/misc/s2_dates_{tag}.hkl",
        f"{base}/clouds/clouds_{tag}.hkl",
    ]


def prediction_key(year: int, x_tile: int, y_tile: int) -> str:
    """Return the expected S3 key for a tile's prediction GeoTIFF."""
    tag = f"{x_tile}X{y_tile}Y"
    return f"{year}/tiles/{x_tile}/{y_tile}/{tag}_FINAL.tif"
