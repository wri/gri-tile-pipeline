"""DuckDB/parquet tile resolution — extracted from handle_inbound_request.py."""

from __future__ import annotations

import json
import re
import sys
from typing import TYPE_CHECKING, Any

import duckdb
import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd

TILE_RE: re.Pattern = re.compile(r"^\s*(?P<x>\-?\d+)X(?P<y>\-?\d+)Y\s*$")


def decode_tile(token: str) -> tuple[int, int] | None:
    """Parse ``'1035X727Y'`` into ``(1035, 727)``."""
    m = TILE_RE.match(token)
    if m is None:
        return None
    return int(m.group("x")), int(m.group("y"))


def load_missing_tiles_with_years(obj: dict[str, Any]) -> list[tuple[int, str]]:
    """Extract ``(year, tile_str)`` pairs from the inbound JSON structure."""
    pairs: set[tuple[int, str]] = set()
    for _project_uuid, years in obj.items():
        if not isinstance(years, dict):
            continue
        for year_key, payload in years.items():
            if not isinstance(payload, dict):
                continue
            missing = payload.get("missing_tiles", [])
            try:
                year_int = int(year_key)
            except (ValueError, TypeError):
                continue
            if isinstance(missing, list):
                for t in missing:
                    if isinstance(t, str):
                        t_strip = t.strip()
                        if t_strip:
                            pairs.add((year_int, t_strip))
    return sorted(pairs)


def tiles_years_to_dataframe(
    pairs: list[tuple[int, str]],
) -> pd.DataFrame:
    """Convert ``(year, tile_str)`` pairs to a DataFrame with ``Year, X_tile, Y_tile``."""
    rows: list[tuple[int, int, int]] = []
    bad: list[tuple[int, str]] = []
    for year, token in pairs:
        parsed = decode_tile(token)
        if parsed is None:
            bad.append((year, token))
        else:
            x, y = parsed
            rows.append((year, x, y))
    if bad:
        examples = ", ".join(f"({yr}, '{tok}')" for yr, tok in bad[:10])
        sys.stderr.write(
            f"[WARN] Skipped {len(bad)} unparseable (year, tile) pairs: "
            + examples
            + ("..." if len(bad) > 10 else "")
            + "\n"
        )
    if not rows:
        return pd.DataFrame(columns=["Year", "X_tile", "Y_tile"])
    return pd.DataFrame(rows, columns=["Year", "X_tile", "Y_tile"]).drop_duplicates()


def resolve_tiles(
    input_json: str,
    parquet_path: str,
    x_col: str = "X_tile",
    y_col: str = "Y_tile",
    limit: int | None = None,
) -> pd.DataFrame:
    """Load a JSON request, resolve tile coordinates via DuckDB parquet join.

    Returns a DataFrame with columns ``Year, X, Y, Y_tile, X_tile``.
    """
    with open(input_json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    tile_year_pairs = load_missing_tiles_with_years(obj)
    if not tile_year_pairs:
        return pd.DataFrame(columns=["Year", "X", "Y", y_col, x_col])

    tiles_df = tiles_years_to_dataframe(tile_year_pairs)
    if tiles_df.empty:
        return pd.DataFrame(columns=["Year", "X", "Y", y_col, x_col])

    con = duckdb.connect()
    try:
        con.register("tiles", tiles_df)
        query = f"""
            SELECT t."Year", p."X", p."Y", p."{y_col}", p."{x_col}"
            FROM read_parquet('{parquet_path}') p
            INNER JOIN tiles t
            ON p."{x_col}" = t."X_tile" AND p."{y_col}" = t."Y_tile"
        """
        if limit is not None and limit > 0:
            query += " LIMIT ?"
            return con.execute(query, [limit]).fetch_df()
        return con.execute(query).fetch_df()
    finally:
        con.close()


def identify_tiles_for_polygons(
    gdf: "gpd.GeoDataFrame",
    lookup_parquet: str = "data/tiledb.parquet",
    lookup_csv: str = "",
) -> list[dict[str, Any]]:
    """Spatial-join polygons against the tile grid to find required tiles.

    Each polygon in *gdf* must have a ``pred_year`` column indicating
    the prediction year for that polygon.

    Returns a deduplicated list of tile dicts with keys:
    ``year``, ``lon``, ``lat``, ``X_tile``, ``Y_tile``.
    """
    from gri_tile_pipeline.zonal.tile_download import load_tile_lookup, pre_filter_tiles

    lookup = load_tile_lookup(parquet_path=lookup_parquet, lookup_csv=lookup_csv or None)

    seen: set[tuple[int, int, int]] = set()
    tiles: list[dict[str, Any]] = []
    for _, row in gdf.iterrows():
        year = int(row["pred_year"])
        pf = pre_filter_tiles(row.geometry, lookup)
        for _, tile_row in pf.iterrows():
            x_tile = int(tile_row["X_tile"])
            y_tile = int(tile_row["Y_tile"])
            key = (year, x_tile, y_tile)
            if key not in seen:
                seen.add(key)
                tiles.append({
                    "year": year,
                    "lon": float(tile_row["X"]),
                    "lat": float(tile_row["Y"]),
                    "X_tile": x_tile,
                    "Y_tile": y_tile,
                })
    return tiles
