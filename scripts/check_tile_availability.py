#!/usr/bin/env python
"""Check TTC tile availability for project polygons or an AOI file.

Supports two input modes:
  1. TerraMatch geoparquet (--geoparquet + --short-name) — per-polygon analysis
  2. Generic AOI file (--aoi) — GeoJSON, GeoPackage, or any GDAL-readable vector

Usage:
    # List available projects (geoparquet mode):
    python scripts/check_tile_availability.py \
        --geoparquet ~/Desktop/experiments/tm-data-test/tm.geoparquet

    # Show intersecting tiles only (no AWS needed):
    python scripts/check_tile_availability.py \
        --geoparquet ~/Desktop/experiments/tm-data-test/tm.geoparquet \
        --short-name ETH_22_SUNARMA \
        --tiles-only

    # Check tile availability on S3:
    python scripts/check_tile_availability.py \
        --geoparquet ~/Desktop/experiments/tm-data-test/tm.geoparquet \
        --short-name ZMB_22_WEFZAMB \
        --year 2024 \
        --aws-profile example-profile

    # AOI mode — show intersecting tiles for a GeoPackage:
    python scripts/check_tile_availability.py \
        --aoi region.gpkg --tiles-only

    # AOI mode — check S3 and export missing tiles as CSV:
    python scripts/check_tile_availability.py \
        --aoi region.geojson --year 2024 \
        --aws-profile example-profile --export-csv missing.csv

    # CSV mode — check S3 for tiles listed in a CSV (supports multi-year):
    python scripts/check_tile_availability.py \
        --csv c2_ttc_missing.csv \
        --aws-profile example-profile --export-csv still_missing.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import duckdb
from loguru import logger

# Add src/ to path so we can import pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Suppress pipeline module logging (we handle our own output)
logger.disable("gri_tile_pipeline")

from gri_tile_pipeline.storage.tile_paths import prediction_key, raw_ard_keys  # noqa: E402
from gri_tile_pipeline.tiles.csv_io import read_tiles_csv, write_tiles_csv  # noqa: E402


HALF_TILE = 1.0 / 36  # half of 1/18 degree tile size


def _make_s3_store(bucket: str, region: str, profile: str | None = None):
    """Build an obstore S3Store with proper credential handling.

    obstore doesn't read AWS_PROFILE natively, so we use
    Boto3CredentialProvider to delegate auth to boto3.
    """
    import boto3
    from obstore.auth.boto3 import Boto3CredentialProvider
    from obstore.store import S3Store

    session = boto3.Session(profile_name=profile)
    credential_provider = Boto3CredentialProvider(session)
    return S3Store(bucket, region=region, credential_provider=credential_provider)


def list_projects(con: duckdb.DuckDBPyConnection, geoparquet: str) -> None:
    """Print available projects with polygon counts."""
    rows = con.execute(
        """
        SELECT short_name, COUNT(*) as cnt, country,
               SUM(CASE WHEN NOT ST_IsValid(geom) THEN 1 ELSE 0 END) as invalid
        FROM read_parquet($1)
        WHERE short_name IS NOT NULL
        GROUP BY short_name, country
        ORDER BY cnt DESC
        """,
        [geoparquet],
    ).fetchall()

    print(f"{'short_name':<30} {'country':>7} {'polygons':>9} {'invalid':>8}")
    print("-" * 58)
    for name, cnt, country, invalid in rows:
        inv_str = str(invalid) if invalid > 0 else ""
        print(f"{name:<30} {country or '':>7} {cnt:>9} {inv_str:>8}")
    print(f"\n{len(rows)} projects with short_name set")


def spatial_join(
    con: duckdb.DuckDBPyConnection,
    geoparquet: str,
    tiledb: str,
    short_name: str,
) -> tuple[dict, dict[str, list[tuple[int, int]]], dict[tuple[int, int], tuple[float, float]], int, int]:
    """Run DuckDB spatial join: polygons x tile grid.

    Returns:
        (poly_info, poly_to_tiles, tile_coords, total_polys, skipped_invalid)
    """
    counts = con.execute(
        """
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN NOT ST_IsValid(geom) THEN 1 ELSE 0 END) as invalid
        FROM read_parquet($1)
        WHERE short_name = $2
        """,
        [geoparquet, short_name],
    ).fetchone()
    total_polys = counts[0]
    skipped_invalid = counts[1]

    if total_polys == 0:
        print(f"No polygons found for short_name = '{short_name}'")
        sys.exit(1)

    rows = con.execute(
        f"""
        WITH polys AS (
            SELECT poly_uuid, poly_name, site_name, geom
            FROM read_parquet($1)
            WHERE short_name = $2
            AND ST_IsValid(geom)
        )
        SELECT
            p.poly_uuid, p.poly_name, p.site_name,
            t.X_tile, t.Y_tile, t.X, t.Y
        FROM polys p, read_parquet($3) t
        WHERE ST_Intersects(
            p.geom,
            ST_MakeEnvelope(
                t.X - {HALF_TILE}, t.Y - {HALF_TILE},
                t.X + {HALF_TILE}, t.Y + {HALF_TILE}
            )
        )
        """,
        [geoparquet, short_name, tiledb],
    ).fetchall()

    poly_to_tiles: dict[str, list[tuple[int, int]]] = defaultdict(list)
    poly_info: dict[str, dict] = {}
    tile_coords: dict[tuple[int, int], tuple[float, float]] = {}
    for poly_uuid, poly_name, site_name, x_tile, y_tile, x, y in rows:
        xt, yt = int(x_tile), int(y_tile)
        poly_to_tiles[poly_uuid].append((xt, yt))
        if poly_uuid not in poly_info:
            poly_info[poly_uuid] = {
                "poly_name": poly_name or "",
                "site_name": site_name or "",
            }
        if (xt, yt) not in tile_coords:
            tile_coords[(xt, yt)] = (float(x), float(y))

    return poly_info, poly_to_tiles, tile_coords, total_polys, skipped_invalid


def spatial_join_aoi(
    con: duckdb.DuckDBPyConnection,
    aoi_path: str,
    tiledb: str,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Spatial join an AOI file (GeoJSON, GeoPackage, etc.) against the tile grid.

    Returns:
        tile_coords: mapping of (X_tile, Y_tile) -> (lon, lat)
    """
    rows = con.execute(
        f"""
        SELECT DISTINCT t.X_tile, t.Y_tile, t.X, t.Y
        FROM ST_Read($1) a, read_parquet($2) t
        WHERE ST_Intersects(
            a.geom,
            ST_MakeEnvelope(
                t.X - {HALF_TILE}, t.Y - {HALF_TILE},
                t.X + {HALF_TILE}, t.Y + {HALF_TILE}
            )
        )
        """,
        [aoi_path, tiledb],
    ).fetchall()

    tile_coords: dict[tuple[int, int], tuple[float, float]] = {}
    for x_tile, y_tile, x, y in rows:
        tile_coords[(int(x_tile), int(y_tile))] = (float(x), float(y))

    if not tile_coords:
        print(f"No tiles intersect the AOI in '{aoi_path}'")
        sys.exit(1)

    return tile_coords


def build_tile_dicts(
    poly_to_tiles: dict[str, list[tuple[int, int]]],
    tile_coords: dict[tuple[int, int], tuple[float, float]],
    year: int,
) -> list[dict]:
    """Build unique tile dicts for check_availability()."""
    unique_tiles: set[tuple[int, int]] = set()
    for tiles in poly_to_tiles.values():
        unique_tiles.update(tiles)

    return [
        {
            "year": year,
            "lon": tile_coords.get((x, y), (0.0, 0.0))[0],
            "lat": tile_coords.get((x, y), (0.0, 0.0))[1],
            "X_tile": x,
            "Y_tile": y,
        }
        for x, y in sorted(unique_tiles)
    ]


def validate_aws(store) -> None:
    """Try a head call on a known prefix to verify credentials work."""
    import obstore as obs

    try:
        # head a nonexistent key — FileNotFoundError means creds work,
        # any other exception means auth/network failure
        obs.head(store, "__credential_check__")
    except FileNotFoundError:
        pass  # expected — credentials are valid, key just doesn't exist
    except Exception as e:
        first_line = str(e).split("\n")[0]
        print("AWS credential check failed:")
        print(f"  {first_line}")
        print()
        print("Make sure you have valid AWS credentials. Options:")
        print("  --aws-profile <profile>   Set an AWS profile")
        print("  aws sso login --profile <name>  If using SSO")
        sys.exit(1)


def print_tiles_only(
    short_name: str,
    poly_to_tiles: dict[str, list[tuple[int, int]]],
    tile_coords: dict[tuple[int, int], tuple[float, float]],
    total_polys: int,
    skipped_invalid: int,
    year: int,
    bucket: str,
    check_type: str,
) -> None:
    """Print intersecting tiles without checking S3."""
    unique_tiles: set[tuple[int, int]] = set()
    tile_poly_count: dict[tuple[int, int], int] = defaultdict(int)
    for tiles in poly_to_tiles.values():
        for tile in tiles:
            unique_tiles.add(tile)
            tile_poly_count[tile] += 1

    valid_polys = total_polys - skipped_invalid
    print(f"Project:      {short_name}")
    print(f"Polygons:     {valid_polys:,}", end="")
    if skipped_invalid > 0:
        print(f" ({skipped_invalid} with invalid geometry, skipped)", end="")
    print()
    print(f"Unique tiles: {len(unique_tiles)}")
    print()

    print("Intersecting tiles:")
    print(f"  {'tile':<14} {'center_lon':>10} {'center_lat':>10} {'polygons':>9}  {'s3_key'}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*9}  {'-'*40}")
    for x, y in sorted(unique_tiles):
        lon, lat = tile_coords.get((x, y), (0.0, 0.0))
        count = tile_poly_count[(x, y)]
        if check_type == "predictions":
            key = prediction_key(year, x, y)
        else:
            key = raw_ard_keys(year, x, y)[0].rsplit("/raw/", 1)[0] + "/"
        print(f"  {x}X{y}Y{'':<6} {lon:>10.4f} {lat:>10.4f} {count:>9}  s3://{bucket}/{key}")


def main():
    parser = argparse.ArgumentParser(
        description="Check TTC tile availability for project polygons, an AOI file, or a tiles CSV."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--geoparquet", help="Path to TerraMatch geoparquet file"
    )
    input_group.add_argument(
        "--aoi", help="Path to AOI file (GeoJSON, GeoPackage, or any GDAL-readable vector)"
    )
    input_group.add_argument(
        "--csv", help="Path to tiles CSV file (Year,X,Y,Y_tile,X_tile) — checks S3 for listed tiles"
    )
    parser.add_argument(
        "--short-name",
        default=None,
        help="Project short_name to filter (geoparquet mode only). Omit to list available projects.",
    )
    parser.add_argument("--year", type=int, default=2024, help="Year to check")
    parser.add_argument(
        "--bucket", default="tof-output", help="S3 bucket (default: tof-output)"
    )
    parser.add_argument(
        "--region", default="us-east-1", help="S3 bucket region (default: us-east-1)"
    )
    parser.add_argument(
        "--check-type",
        choices=["predictions", "raw_ard"],
        default="predictions",
        help="What to check for (default: predictions)",
    )
    parser.add_argument(
        "--aws-profile", default=None, help="AWS profile name for S3 access"
    )
    parser.add_argument(
        "--tiledb",
        default="data/tiledb.parquet",
        help="Path to tiledb.parquet (default: data/tiledb.parquet)",
    )
    parser.add_argument(
        "--tiles-only",
        action="store_true",
        help="Only show intersecting tiles, skip S3 availability check (no AWS needed)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all polygons in coverage table (default: only incomplete when >30 polygons)",
    )
    parser.add_argument(
        "--export-csv",
        default=None,
        metavar="PATH",
        help="Export missing tiles (or all tiles if --tiles-only) to CSV (Year,X,Y,Y_tile,X_tile)",
    )
    args = parser.parse_args()

    # Validate argument combinations
    if args.aoi and args.short_name:
        parser.error("--short-name is only used with --geoparquet, not --aoi")
    if args.csv and args.short_name:
        parser.error("--short-name is only used with --geoparquet, not --csv")
    if args.csv and args.tiles_only:
        parser.error("--tiles-only is redundant with --csv (tiles are already known)")

    # --- CSV mode ---
    if args.csv:
        tile_dicts = read_tiles_csv(args.csv)
        if not tile_dicts:
            print(f"No tiles found in '{args.csv}'")
            sys.exit(1)

        # Summarise by year
        year_counts: dict[int, int] = {}
        for t in tile_dicts:
            year_counts[t["year"]] = year_counts.get(t["year"], 0) + 1

        print(f"Input:        {args.csv}")
        print(f"Total tiles:  {len(tile_dicts)}")
        print(f"Years:        {', '.join(f'{y} ({c})' for y, c in sorted(year_counts.items()))}")
        print(f"Check type:   {args.check_type}")
        print(f"S3 bucket:    s3://{args.bucket}")
        print()

        print(f"Connecting to s3://{args.bucket} (region={args.region}) ...")
        store = _make_s3_store(args.bucket, args.region, args.aws_profile)
        validate_aws(store)
        print("  OK\n")

        print("Checking tile availability on S3 ...")
        from gri_tile_pipeline.tiles.availability import check_availability

        result = check_availability(
            tile_dicts, f"s3://{args.bucket}",
            check_type=args.check_type, region=args.region,
            store=store,
        )
        existing = result["existing"]
        missing = result["missing"]

        existing_set = {(t["year"], t["X_tile"], t["Y_tile"]) for t in existing}
        missing_set = {(t["year"], t["X_tile"], t["Y_tile"]) for t in missing}

        total = len(tile_dicts)
        n_existing = len(existing_set)
        pct = (n_existing / total * 100) if total > 0 else 0

        print("\nTile Availability:")
        print(f"  Existing: {n_existing} / {total} ({pct:.1f}%)")
        print(f"  Missing:  {len(missing_set)}")

        if missing_set:
            print("\nMissing tiles:")
            for year, x, y in sorted(missing_set):
                if args.check_type == "predictions":
                    key = prediction_key(year, x, y)
                else:
                    key = raw_ard_keys(year, x, y)[0].rsplit("/", 1)[0] + "/"
                print(f"  {year}  {x}X{y}Y  s3://{args.bucket}/{key}")

        if args.export_csv:
            write_tiles_csv(args.export_csv, missing)
            print(f"\nExported {len(missing)} missing tiles to {args.export_csv}")
        return

    # Set up DuckDB with spatial extension
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    # --- AOI mode ---
    if args.aoi:
        print(f"Running spatial join for AOI '{args.aoi}' ...")
        tile_coords = spatial_join_aoi(con, args.aoi, args.tiledb)
        con.close()

        tile_dicts = [
            {"year": args.year, "lon": lon, "lat": lat, "X_tile": x, "Y_tile": y}
            for (x, y), (lon, lat) in sorted(tile_coords.items())
        ]

        if args.tiles_only:
            print(f"AOI:          {args.aoi}")
            print(f"Unique tiles: {len(tile_coords)}")
            print()
            print("Intersecting tiles:")
            print(f"  {'tile':<14} {'center_lon':>10} {'center_lat':>10}  {'s3_key'}")
            print(f"  {'-'*14} {'-'*10} {'-'*10}  {'-'*40}")
            for (x, y), (lon, lat) in sorted(tile_coords.items()):
                if args.check_type == "predictions":
                    key = prediction_key(args.year, x, y)
                else:
                    key = raw_ard_keys(args.year, x, y)[0].rsplit("/raw/", 1)[0] + "/"
                print(f"  {x}X{y}Y{'':<6} {lon:>10.4f} {lat:>10.4f}  s3://{args.bucket}/{key}")
            if args.export_csv:
                write_tiles_csv(args.export_csv, tile_dicts)
                print(f"\nExported {len(tile_dicts)} tiles to {args.export_csv}")
            return

        # S3 availability check
        print(f"Connecting to s3://{args.bucket} (region={args.region}) ...")
        store = _make_s3_store(args.bucket, args.region, args.aws_profile)
        validate_aws(store)
        print("  OK\n")

        print(f"AOI:          {args.aoi}")
        print(f"Unique tiles: {len(tile_coords)}")
        print(f"Year:         {args.year}")
        print(f"Check type:   {args.check_type}")
        print(f"S3 bucket:    s3://{args.bucket}")
        print()

        print("Checking tile availability on S3 ...")
        from gri_tile_pipeline.tiles.availability import check_availability

        result = check_availability(
            tile_dicts, f"s3://{args.bucket}",
            check_type=args.check_type, region=args.region,
            store=store,
        )
        existing_set = {(t["X_tile"], t["Y_tile"]) for t in result["existing"]}
        missing = result["missing"]
        missing_set = {(t["X_tile"], t["Y_tile"]) for t in missing}

        total = len(tile_coords)
        n_existing = len(existing_set)
        pct = (n_existing / total * 100) if total > 0 else 0

        print("\nTile Availability:")
        print(f"  Existing: {n_existing} / {total} ({pct:.1f}%)")
        print(f"  Missing:  {len(missing_set)}")

        if missing_set:
            print("\nMissing tiles:")
            for x, y in sorted(missing_set):
                if args.check_type == "predictions":
                    key = prediction_key(args.year, x, y)
                else:
                    key = raw_ard_keys(args.year, x, y)[0].rsplit("/", 1)[0] + "/"
                print(f"  {x}X{y}Y  s3://{args.bucket}/{key}")

        if args.export_csv:
            write_tiles_csv(args.export_csv, missing)
            print(f"\nExported {len(missing)} missing tiles to {args.export_csv}")
        return

    # --- Geoparquet mode ---

    # If no short_name, list available projects and exit
    if args.short_name is None:
        list_projects(con, args.geoparquet)
        con.close()
        return

    # Run spatial join
    print(f"Running spatial join for project '{args.short_name}' ...")
    poly_info, poly_to_tiles, tile_coords, total_polys, skipped_invalid = spatial_join(
        con, args.geoparquet, args.tiledb, args.short_name
    )
    con.close()

    # --tiles-only: just print tile list and exit
    if args.tiles_only:
        print_tiles_only(
            args.short_name, poly_to_tiles, tile_coords,
            total_polys, skipped_invalid, args.year, args.bucket, args.check_type,
        )
        if args.export_csv:
            all_tile_dicts = build_tile_dicts(poly_to_tiles, tile_coords, args.year)
            write_tiles_csv(args.export_csv, all_tile_dicts)
            print(f"\nExported {len(all_tile_dicts)} tiles to {args.export_csv}")
        return

    # Build S3 store with boto3 credential provider
    print(f"Connecting to s3://{args.bucket} (region={args.region}) ...")
    store = _make_s3_store(args.bucket, args.region, args.aws_profile)
    validate_aws(store)
    print("  OK\n")

    tile_dicts = build_tile_dicts(poly_to_tiles, tile_coords, args.year)
    unique_tiles = {(t["X_tile"], t["Y_tile"]) for t in tile_dicts}

    # Count polygons per tile
    tile_poly_count: dict[tuple[int, int], int] = defaultdict(int)
    for tiles in poly_to_tiles.values():
        for tile in tiles:
            tile_poly_count[tile] += 1

    # Header
    valid_polys = total_polys - skipped_invalid
    print(f"Project:      {args.short_name}")
    print(f"Polygons:     {valid_polys:,}", end="")
    if skipped_invalid > 0:
        print(f" ({skipped_invalid} with invalid geometry, skipped)", end="")
    print()
    print(f"Unique tiles: {len(unique_tiles)}")
    print(f"Year:         {args.year}")
    print(f"Check type:   {args.check_type}")
    print(f"S3 bucket:    s3://{args.bucket}")
    print()

    # Check availability on S3 — pass the boto3-backed store
    print("Checking tile availability on S3 ...")
    from gri_tile_pipeline.tiles.availability import check_availability

    result = check_availability(
        tile_dicts, f"s3://{args.bucket}",
        check_type=args.check_type, region=args.region,
        store=store,
    )
    existing = result["existing"]
    missing = result["missing"]

    existing_set = {(t["X_tile"], t["Y_tile"]) for t in existing}
    missing_set = {(t["X_tile"], t["Y_tile"]) for t in missing}

    total = len(unique_tiles)
    n_existing = len(existing_set)
    pct = (n_existing / total * 100) if total > 0 else 0

    print("\nTile Availability:")
    print(f"  Existing: {n_existing} / {total} ({pct:.1f}%)")
    print(f"  Missing:  {len(missing_set)}")

    if missing_set:
        print("\nMissing tiles:")
        for x, y in sorted(missing_set):
            count = tile_poly_count.get((x, y), 0)
            s = "s" if count != 1 else ""
            if args.check_type == "predictions":
                key = prediction_key(args.year, x, y)
            else:
                key = raw_ard_keys(args.year, x, y)[0].rsplit("/", 1)[0] + "/"
            print(f"  {x}X{y}Y  (covers {count} polygon{s})  s3://{args.bucket}/{key}")

    if args.export_csv:
        write_tiles_csv(args.export_csv, missing)
        print(f"\nExported {len(missing)} missing tiles to {args.export_csv}")

    # Per-polygon coverage analysis
    fully_covered = 0
    incomplete_rows: list[tuple[str, str, int, int, int]] = []
    for poly_uuid in sorted(poly_to_tiles.keys()):
        tiles = poly_to_tiles[poly_uuid]
        tile_set = set(tiles)
        n_avail = len(tile_set & existing_set)
        n_miss = len(tile_set & missing_set)
        if n_miss == 0:
            fully_covered += 1
        else:
            info = poly_info.get(poly_uuid, {})
            site = info.get("site_name", "")[:25]
            incomplete_rows.append((poly_uuid, site, len(tile_set), n_avail, n_miss))

    n_polys = len(poly_to_tiles)
    show_all = args.verbose or n_polys <= 30

    if show_all:
        # Full table
        print("\nPer-polygon coverage:")
        print(f"  {'poly_uuid':<40} {'site_name':<25} {'tiles':>5} {'avail':>5} {'miss':>5}")
        print(f"  {'-'*40} {'-'*25} {'-'*5} {'-'*5} {'-'*5}")
        for poly_uuid in sorted(poly_to_tiles.keys()):
            tiles = poly_to_tiles[poly_uuid]
            tile_set = set(tiles)
            n_avail = len(tile_set & existing_set)
            n_miss = len(tile_set & missing_set)
            info = poly_info.get(poly_uuid, {})
            site = info.get("site_name", "")[:25]
            mark = "+" if n_miss == 0 else "X"
            print(
                f"  {poly_uuid:<40} {site:<25} {len(tile_set):>5} {n_avail:>5} {n_miss:>5}  {mark}"
            )
    elif incomplete_rows:
        # Only show incomplete polygons
        print(f"\nPolygons with missing tiles ({len(incomplete_rows)}):")
        print(f"  {'poly_uuid':<40} {'site_name':<25} {'tiles':>5} {'avail':>5} {'miss':>5}")
        print(f"  {'-'*40} {'-'*25} {'-'*5} {'-'*5} {'-'*5}")
        for poly_uuid, site, n_tiles, n_avail, n_miss in incomplete_rows:
            print(f"  {poly_uuid:<40} {site:<25} {n_tiles:>5} {n_avail:>5} {n_miss:>5}  X")
        print(f"  ({fully_covered:,} fully covered polygons not shown, use --verbose to see all)")

    pct_covered = (fully_covered / n_polys * 100) if n_polys else 0
    print(
        f"\nSummary: {fully_covered:,} / {n_polys:,} polygons "
        f"fully covered ({pct_covered:.1f}%)"
    )


if __name__ == "__main__":
    main()
