"""
Handle an inbound request in the form of a JSON object.

Structure looks like this:
{
    "uuid":{ # project-level uuid
        "YYYY":{
            "all_tiles": [], # ignore
            "missing_tiles": [] # process all these
        }
    }
}


The goal is to parse the json object, take all the "missing_tiles" keys, dedupe the values, then perform a lookup
of the values in these keys against a tiledb.parquet file using duckdb in python.

Example:

python scripts/handle_inbound_request.py data/ppc_2025_batch2_2025_ttc_tile_avail_2025-08-25.json --parquet data/tiledb.parquet > data/matches.csv

Output is a csv Year,X,Y,Y_tile,X_tile

"""

import argparse
import json
import os
import re
import sys
from typing import Iterable, List, Set, Tuple, Dict, Any, Optional

import duckdb
import pandas as pd

# Regex to parse tile ids like '1035X727Y'
TILE_RE = re.compile(r"^\s*(?P<x>\-?\d+)X(?P<y>\-?\d+)Y\s*$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Parse JSON of project-year missing tiles and return all matching rows "
            "from a TileDB parquet index using DuckDB."
        )
    )
    p.add_argument("input_json", help="Path to input JSON file.")
    p.add_argument(
        "--parquet",
        default="data/tiledb.parquet",
        help="Path to the TileDB parquet file (default: data/tiledb.parquet).",
    )
    p.add_argument(
        "--columns",
        default=None,
        help=(
            "Optional comma-separated list of columns to return from the parquet. "
            "Defaults to all columns."
        ),
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Optional output file. If provided, writes CSV to this path. "
            "Otherwise prints to stdout."
        ),
    )
    p.add_argument(
        "--x-col",
        default="X_tile",
        help="Name of the X tile column in the parquet (default: X_tile).",
    )
    p.add_argument(
        "--y-col",
        default="Y_tile",
        help="Name of the Y tile column in the parquet (default: Y_tile).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional LIMIT for the result set (debugging).",
    )
    return p.parse_args()


def load_missing_tiles(obj: Dict[str, Any]) -> Set[str]:
    """
    Traverse the nested project->year structure and collect all 'missing_tiles' values.
    Deduplicates by returning a set of tile strings like '1035X727Y'.
    """
    tiles: Set[str] = set()
    for project_uuid, years in obj.items():
        if not isinstance(years, dict):
            continue
        for year, payload in years.items():
            if not isinstance(payload, dict):
                continue
            missing = payload.get("missing_tiles", [])
            if isinstance(missing, list):
                for t in missing:
                    if isinstance(t, str):
                        t_strip = t.strip()
                        if t_strip:
                            tiles.add(t_strip)
    return tiles


def decode_tile(token: str) -> Optional[Tuple[int, int]]:
    """
    Convert '1035X727Y' -> (1035, 727). Returns None if not matched.
    """
    m = TILE_RE.match(token)
    if not m:
        return None
    return (int(m.group("x")), int(m.group("y")))


def tiles_to_dataframe(tiles: Iterable[str]) -> pd.DataFrame:
    rows: List[Tuple[int, int]] = []
    bad: List[str] = []
    for t in tiles:
        parsed = decode_tile(t)
        if parsed is None:
            bad.append(t)
        else:
            rows.append(parsed)
    if bad:
        sys.stderr.write(
            f"[WARN] Skipped {len(bad)} unparseable tiles: "
            + ", ".join(bad[:10])
            + ("..." if len(bad) > 10 else "")
            + "\n"
        )
    if not rows:
        return pd.DataFrame(columns=["X_tile", "Y_tile"])
    df = pd.DataFrame(rows, columns=["X_tile", "Y_tile"]).drop_duplicates()
    return df


def load_missing_tiles_with_years(obj: Dict[str, Any]) -> List[Tuple[int, str]]:
    """
    Traverse the nested project->year structure and collect all (year, tile) pairs
    for 'missing_tiles'. Returns a list of tuples like (2022, '1035X727Y'),
    de-duplicated at the pair level.
    """
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
            except Exception:
                # Skip non-integer year keys
                continue
            if isinstance(missing, list):
                for t in missing:
                    if isinstance(t, str):
                        t_strip = t.strip()
                        if t_strip:
                            pairs.add((year_int, t_strip))
    # Return a stable order (not required, but nice for reproducibility)
    return sorted(pairs)


def tiles_years_to_dataframe(pairs: Iterable[Tuple[int, str]]) -> pd.DataFrame:
    """
    Convert iterable of (year, tile_str) into a DataFrame with columns
    ["Year", "X_tile", "Y_tile"]. Invalid tiles are skipped with a warning.
    """
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
        # Show a few examples to stderr
        examples = ", ".join([f"({yr}, '{tok}')" for yr, tok in bad[:10]])
        sys.stderr.write(
            f"[WARN] Skipped {len(bad)} unparseable (year, tile) pairs: "
            + examples
            + ("..." if len(bad) > 10 else "")
            + "\n"
        )
    if not rows:
        return pd.DataFrame(columns=["Year", "X_tile", "Y_tile"])
    df = pd.DataFrame(rows, columns=["Year", "X_tile", "Y_tile"]).drop_duplicates()
    return df


def main() -> None:
    args = parse_args()

    # Load JSON
    try:
        with open(args.input_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to read JSON: {e}\n")
        sys.exit(2)

    # Extract & parse tiles with years
    tile_year_pairs = load_missing_tiles_with_years(obj)
    if not tile_year_pairs:
        sys.stderr.write("[INFO] No missing_tiles found in input.\n")
        # Exit gracefully with empty output
        if args.output:
            pd.DataFrame().to_csv(args.output, index=False)
        else:
            # Print nothing
            pass
        return

    tiles_df = tiles_years_to_dataframe(tile_year_pairs)
    if tiles_df.empty:
        sys.stderr.write("[INFO] No valid (year, tile) pairs after parsing.\n")
        if args.output:
            pd.DataFrame().to_csv(args.output, index=False)
        return

    # Connect DuckDB
    con = duckdb.connect()

    # Register the tiles dataframe as a relation for efficient joining
    con.register("tiles", tiles_df)

    # Compose query using a join on (X_tile, Y_tile).
    # Column names in parquet can be customized via --x-col/--y-col.
    parquet_path = args.parquet
    xcol = args.x_col
    ycol = args.y_col

    # We use parameterized SQL for LIMIT only; column/table identifiers need string formatting.
    base_query = f"""
        SELECT t."Year", p."X", p."Y", p."{ycol}", p."{xcol}"
        FROM read_parquet('{parquet_path}') p
        INNER JOIN tiles t
        ON p."{xcol}" = t."X_tile" AND p."{ycol}" = t."Y_tile"
    """
    if args.limit is not None and args.limit > 0:
        base_query += " LIMIT ?"

    try:
        if args.limit is not None and args.limit > 0:
            result_df = con.execute(base_query, [args.limit]).fetch_df()
        else:
            result_df = con.execute(base_query).fetch_df()
    except Exception as e:
        sys.stderr.write(f"[ERROR] DuckDB query failed: {e}\n")
        sys.exit(3)
    finally:
        con.close()

    # Output
    if args.output:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        result_df.to_csv(args.output, index=False)
        print(args.output)
    else:
        # Print CSV to stdout
        result_df.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
