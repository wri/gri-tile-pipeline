"""DuckDB/parquet tile resolution — extracted from handle_inbound_request.py."""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb
import pandas as pd

TILE_RE = re.compile(r"^\s*(?P<x>\-?\d+)X(?P<y>\-?\d+)Y\s*$")


def decode_tile(token: str) -> Optional[Tuple[int, int]]:
    """Parse ``'1035X727Y'`` into ``(1035, 727)``."""
    m = TILE_RE.match(token)
    if m is None:
        return None
    return int(m.group("x")), int(m.group("y"))


def load_missing_tiles_with_years(obj: Dict[str, Any]) -> List[Tuple[int, str]]:
    """Extract ``(year, tile_str)`` pairs from the inbound JSON structure."""
    pairs: Set[Tuple[int, str]] = set()
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
    pairs: List[Tuple[int, str]],
) -> pd.DataFrame:
    """Convert ``(year, tile_str)`` pairs to a DataFrame with ``Year, X_tile, Y_tile``."""
    rows: List[Tuple[int, int, int]] = []
    bad: List[Tuple[int, str]] = []
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
    limit: Optional[int] = None,
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
