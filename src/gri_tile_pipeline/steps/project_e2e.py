"""End-to-end pipeline for a TerraMatch project: short_name -> zonal stats CSV.

Chains: extract polygons -> survey TTC -> spatial join tiles ->
check availability -> download ARD -> predict -> zonal stats -> CSV.

Supports two input modes:
  - **short_name**: select all polygons for a project by name
  - **request CSV**: select polygons by (project_id, plantstart_year) pairs
"""

from __future__ import annotations

import csv as csv_mod
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import pandas as pd
from loguru import logger

from gri_tile_pipeline.config import PipelineConfig
from gri_tile_pipeline.exit_codes import ExitCode, exit_code_from_tracker


# ---------------------------------------------------------------------------
# Step 1: Extract project polygons from geoparquet
# ---------------------------------------------------------------------------

def _gdf_from_duckdb(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert a DuckDB result (with ``__wkb`` column) to a GeoDataFrame."""
    from shapely import from_wkb

    geoms = []
    bad_idx = []
    uuid_col = "poly_uuid" if "poly_uuid" in df.columns else None
    for i, wkb in enumerate(df["__wkb"]):
        try:
            geoms.append(from_wkb(bytes(wkb)))
        except Exception as e:
            uid = df.iloc[i][uuid_col] if uuid_col else f"row {i}"
            logger.warning(f"Skipping invalid geometry for {uid}: {e}")
            geoms.append(None)
            bad_idx.append(i)

    df["geometry"] = geoms
    if bad_idx:
        logger.warning(f"Dropped {len(bad_idx)} polygon(s) with invalid geometry")
        df = df.drop(index=df.index[bad_idx]).reset_index(drop=True)

    drop = [c for c in ("geom", "__wkb") if c in df.columns]
    return gpd.GeoDataFrame(df.drop(columns=drop), geometry="geometry", crs="EPSG:4326")


def _ttc_coverage(gdf: gpd.GeoDataFrame) -> tuple[int, int]:
    """Count polygons with TTC and with correct-year TTC."""
    ttc_col = "ttc" if "ttc" in gdf.columns else None
    n_with, n_correct = 0, 0
    if ttc_col:
        for _, row in gdf.iterrows():
            ttc = row[ttc_col]
            pred_yr = row["pred_year"]
            if pd.notna(ttc) and isinstance(ttc, (dict, list)) and len(ttc) > 0:
                n_with += 1
                if pred_yr in ttc:
                    n_correct += 1
    return n_with, n_correct


def _extract_project(
    short_name: str,
    geoparquet: str,
    year_override: int | None = None,
) -> tuple[gpd.GeoDataFrame, Path, dict]:
    """Query *geoparquet* for *short_name*, add ``pred_year`` column.

    Returns:
        (gdf_with_pred_year, geojson_path, metadata_dict)
    """
    import duckdb

    con = duckdb.connect()
    con.install_extension("spatial")
    con.load_extension("spatial")
    con.execute("SET geometry_always_xy=true")

    df = con.sql(f"""
        SELECT *, ST_AsWKB(geom) AS __wkb
        FROM read_parquet('{geoparquet}')
        WHERE short_name = '{short_name}'
    """).df()

    if df.empty:
        raise ValueError(f"No polygons found for short_name='{short_name}'")

    gdf = _gdf_from_duckdb(df)

    # Derive per-polygon prediction year
    if year_override is not None:
        gdf["pred_year"] = year_override
    else:
        def _derive_year(ps):
            if ps is not None and hasattr(ps, "year"):
                return ps.year - 1
            return None
        gdf["pred_year"] = gdf["plantstart"].apply(_derive_year)

    invalid = gdf["pred_year"].isna()
    if invalid.all():
        raise ValueError(f"No valid plantstart dates for {short_name}")
    if invalid.any():
        logger.warning(
            f"{invalid.sum()} polygon(s) have no valid plantstart — dropping"
        )
        gdf = gdf[~invalid].copy()

    gdf["pred_year"] = gdf["pred_year"].astype(int)
    year_counts = Counter(gdf["pred_year"])
    n_with_ttc, n_correct_yr = _ttc_coverage(gdf)

    metadata = {
        "label": short_name,
        "n_polygons": len(gdf),
        "country": gdf["country"].iloc[0] if "country" in gdf.columns else "unknown",
        "framework_key": gdf["framework_key"].iloc[0] if "framework_key" in gdf.columns else "unknown",
        "year_counts": dict(year_counts),
        "n_with_ttc": n_with_ttc,
        "n_correct_yr": n_correct_yr,
        "n_missing_ttc": len(gdf) - n_with_ttc,
    }

    out = Path(tempfile.mkdtemp(prefix="ttc_project_")) / f"{short_name}.geojson"
    gdf.to_file(out, driver="GeoJSON")

    return gdf, out, metadata


def _extract_from_request_csv(
    input_csv: str,
    geoparquet: str,
) -> tuple[gpd.GeoDataFrame, Path, dict]:
    """Extract polygons matching ``(project_id, plantstart_year)`` pairs.

    The CSV must have columns ``project_id`` and ``plantstart_year``.
    Prediction year for each polygon = ``plantstart_year - 1``.

    Returns:
        (gdf_with_pred_year, geojson_path, metadata_dict)
    """
    import duckdb

    # Read request CSV
    req = pd.read_csv(input_csv)
    required = {"project_id", "plantstart_year"}
    missing = required - set(req.columns)
    if missing:
        raise ValueError(f"Request CSV missing columns: {sorted(missing)}")

    pairs = list(req[["project_id", "plantstart_year"]].itertuples(index=False, name=None))
    if not pairs:
        raise ValueError("Request CSV is empty")

    logger.info(f"Request CSV: {len(pairs)} (project_id, plantstart_year) pairs")

    con = duckdb.connect()
    con.install_extension("spatial")
    con.load_extension("spatial")
    con.execute("SET geometry_always_xy=true")

    # Build WHERE clause matching (project_id, YEAR(plantstart)) pairs
    conditions = " OR ".join(
        f"(project_id = '{pid}' AND YEAR(plantstart) = {int(yr)})"
        for pid, yr in pairs
    )

    df = con.sql(f"""
        SELECT *, ST_AsWKB(geom) AS __wkb, YEAR(plantstart) AS __ps_year
        FROM read_parquet('{geoparquet}')
        WHERE {conditions}
    """).df()

    if df.empty:
        raise ValueError(
            "No polygons found matching the request CSV. "
            "Check project_id values and plantstart_year ranges."
        )

    gdf = _gdf_from_duckdb(df)

    # pred_year = plantstart_year - 1 (from the actual plantstart, not the CSV)
    gdf["pred_year"] = (gdf["__ps_year"] - 1).astype(int)
    if "__ps_year" in gdf.columns:
        gdf = gdf.drop(columns=["__ps_year"])

    year_counts = Counter(gdf["pred_year"])
    n_with_ttc, n_correct_yr = _ttc_coverage(gdf)

    # Build a human-readable label from project_ids
    project_ids = req["project_id"].unique()
    n_projects = len(project_ids)
    label = f"{n_projects}_project{'s' if n_projects > 1 else ''}_request"

    # Log per-project breakdown
    if "project_id" in gdf.columns:
        for pid in project_ids:
            pid_gdf = gdf[gdf["project_id"] == pid]
            if pid_gdf.empty:
                logger.warning(f"  {pid[:12]}...: 0 polygons matched")
                continue
            sname = pid_gdf["short_name"].iloc[0] if "short_name" in pid_gdf.columns else None
            display = sname if pd.notna(sname) and sname else pid[:12] + "..."
            pid_years = sorted(pid_gdf["pred_year"].unique())
            logger.info(f"  {display}: {len(pid_gdf)} polygons, pred_years={pid_years}")

    metadata = {
        "label": label,
        "n_polygons": len(gdf),
        "country": gdf["country"].iloc[0] if "country" in gdf.columns else "mixed",
        "framework_key": gdf["framework_key"].iloc[0] if "framework_key" in gdf.columns else "mixed",
        "year_counts": dict(year_counts),
        "n_with_ttc": n_with_ttc,
        "n_correct_yr": n_correct_yr,
        "n_missing_ttc": len(gdf) - n_with_ttc,
        "n_projects": n_projects,
    }

    out = Path(tempfile.mkdtemp(prefix="ttc_request_")) / "request_polygons.geojson"
    gdf.to_file(out, driver="GeoJSON")

    return gdf, out, metadata


def _extract_by_filter(
    geoparquet: str,
    *,
    where_sql: str | None = None,
    poly_uuids: list[str] | None = None,
    cohorts: list[str] | None = None,
    project_ids: list[str] | None = None,
    short_names: list[str] | None = None,
    framework_keys: list[str] | None = None,
    year_override: int | None = None,
) -> tuple[gpd.GeoDataFrame, Path, dict]:
    """Extract polygons from *geoparquet* using any combination of filters.

    Used by ``run-project`` when the user provides ``--where`` / ``--poly-uuid`` /
    ``--cohort`` rather than a short_name or request CSV. Re-uses the same
    filter builder as the ``report`` command so behaviour stays consistent.
    """
    from gri_tile_pipeline.duckdb_utils import connect_with_spatial
    from gri_tile_pipeline.reporting.status_report import _build_filter

    poly_uuids = poly_uuids or []
    cohorts = cohorts or []
    project_ids = project_ids or []
    short_names = short_names or []
    framework_keys = framework_keys or []

    con = connect_with_spatial()
    try:
        where, params = _build_filter(
            con, geoparquet, None,
            project_ids, short_names, framework_keys,
            poly_uuids=poly_uuids, cohorts=cohorts, where_sql=where_sql,
        )
        df = con.execute(
            f"SELECT *, ST_AsWKB(geom) AS __wkb FROM read_parquet($1) p WHERE {where}",
            params,
        ).df()
    finally:
        con.close()

    if df.empty:
        raise ValueError(
            "No polygons matched the filter. Check the filter values "
            "(where/poly_uuid/cohort/short_name/project_id/framework_key)."
        )

    gdf = _gdf_from_duckdb(df)

    if year_override is not None:
        gdf["pred_year"] = year_override
    else:
        def _derive_year(ps):
            if ps is not None and hasattr(ps, "year"):
                return ps.year - 1
            return None
        gdf["pred_year"] = gdf["plantstart"].apply(_derive_year)

    invalid = gdf["pred_year"].isna()
    if invalid.all():
        raise ValueError(
            "No polygons have a valid plantstart — pass --year to override."
        )
    if invalid.any():
        logger.warning(f"{invalid.sum()} polygon(s) have no valid plantstart — dropping")
        gdf = gdf[~invalid].copy()
    gdf["pred_year"] = gdf["pred_year"].astype(int)

    year_counts = Counter(gdf["pred_year"])
    n_with_ttc, n_correct_yr = _ttc_coverage(gdf)

    # Human-readable label describing the filter.
    label_parts = []
    if short_names:
        label_parts.append("+".join(short_names) if len(short_names) <= 3 else f"{len(short_names)}sn")
    if project_ids:
        label_parts.append(f"{len(project_ids)}pid")
    if framework_keys:
        label_parts.append("+".join(framework_keys))
    if poly_uuids:
        label_parts.append(f"{len(poly_uuids)}uuids")
    if cohorts:
        label_parts.append("+".join(cohorts))
    if where_sql:
        label_parts.append("where")
    label = "_".join(label_parts) or "filter"

    metadata = {
        "label": label,
        "n_polygons": len(gdf),
        "country": gdf["country"].iloc[0] if "country" in gdf.columns and len(gdf) else "mixed",
        "framework_key": (
            gdf["framework_key"].iloc[0]
            if "framework_key" in gdf.columns and len(gdf) else "mixed"
        ),
        "year_counts": dict(year_counts),
        "n_with_ttc": n_with_ttc,
        "n_correct_yr": n_correct_yr,
        "n_missing_ttc": len(gdf) - n_with_ttc,
        "filter": {
            "where_sql": where_sql,
            "poly_uuids": poly_uuids,
            "cohorts": cohorts,
            "project_ids": project_ids,
            "short_names": short_names,
            "framework_keys": framework_keys,
        },
    }

    out = Path(tempfile.mkdtemp(prefix="ttc_filter_")) / f"{label}.geojson"
    gdf.to_file(out, driver="GeoJSON")

    return gdf, out, metadata


# ---------------------------------------------------------------------------
# Step 3: Identify required tiles via per-polygon spatial join
# ---------------------------------------------------------------------------

def _identify_tiles(
    gdf: gpd.GeoDataFrame,
    lookup_parquet: str = "data/tiledb.parquet",
    lookup_csv: str = "",
) -> List[Dict[str, Any]]:
    """Per-polygon spatial join: for each polygon, find intersecting tiles
    and tag them with that polygon's ``pred_year``.

    Returns deduplicated list of tile dicts (year, lon, lat, X_tile, Y_tile).
    """
    from gri_tile_pipeline.tiles.tile_lookup import identify_tiles_for_polygons

    return identify_tiles_for_polygons(gdf, lookup_parquet=lookup_parquet, lookup_csv=lookup_csv)


# ---------------------------------------------------------------------------
# Helper: resolve tiles bucket from dest
# ---------------------------------------------------------------------------

def _resolve_tiles_bucket(dest: str) -> str:
    """Extract tiles_bucket value from dest for zonal stats.

    For S3: returns bucket name (e.g. 'tof-output')
    For local: returns the local path as-is
    """
    if dest.startswith("s3://"):
        return dest.replace("s3://", "").split("/")[0]
    return dest


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_project_pipeline(
    short_name: str | None,
    dest: str,
    cfg: PipelineConfig,
    *,
    input_csv: str | None = None,
    geoparquet: str = "temp/tm.geoparquet",
    year: int | None = None,
    output: str | None = None,
    local: bool = False,
    max_workers: int = 1,
    skip_existing: bool = True,
    lulc_raster: str | None = None,
    missing_only: bool = False,
    check_only: bool = False,
    dry_run: bool = False,
    where_sql: str | None = None,
    poly_uuids: list[str] | None = None,
    cohorts: list[str] | None = None,
    project_ids: list[str] | None = None,
    short_names: list[str] | None = None,
    framework_keys: list[str] | None = None,
) -> dict:
    """Run the full pipeline for a TerraMatch project.

    Accepts either a *short_name* or an *input_csv* with
    ``(project_id, plantstart_year)`` pairs.

    Steps:
        1. Extract polygons from geoparquet (per-polygon prediction years)
        2. Report TTC coverage status
        3. Spatial join to identify required tiles (per-polygon year)
        4. Check for existing predictions
        5. Download ARD for missing tiles
        6. Run inference for missing tiles
        7. Run zonal statistics (per-year groups)
        8. Output results CSV

    Args:
        short_name: Project identifier in tm.geoparquet. Mutually exclusive
            with *input_csv*.
        dest: S3 URI (s3://bucket) or local path for ARD + predictions.
        cfg: Pipeline configuration.
        input_csv: Path to request CSV with ``project_id, plantstart_year``
            columns. Mutually exclusive with *short_name*.
        geoparquet: Path to TerraMatch geoparquet file.
        year: Override prediction year for ALL polygons (default: per-polygon
            plantstart - 1). Only applies when using *short_name* mode.
        output: Output CSV path (default: temp/{label}_stats.csv). When
            *check_only* is True, this is where the missing tiles CSV is
            written.
        max_workers: Parallel local workers for download/predict.
        skip_existing: Skip tiles already available at dest.
        lulc_raster: Path/URI to LULC raster for error propagation.
        missing_only: Only process polygons that are missing TTC for
            their prediction year. Matches the behaviour of
            ``ttc_status_report.py``.
        check_only: Stop after step 4 and write missing tiles CSV to *output*.
        dry_run: Show plan without executing download/predict/stats.

    Returns:
        Summary dict with keys: label, n_polygons, n_tiles,
        n_tiles_generated, output_path, exit_code.
    """
    has_filter = bool(
        where_sql or poly_uuids or cohorts
        or project_ids or short_names or framework_keys
    )
    sources_provided = sum([bool(short_name), bool(input_csv), has_filter])
    if sources_provided == 0:
        raise ValueError(
            "Provide short_name, input_csv, or at least one filter "
            "(--where / --poly-uuid / --cohort / --project-id / --short-name / --framework-key)"
        )
    if sources_provided > 1:
        raise ValueError(
            "Provide exactly one of: short_name, input_csv, or filter flags"
        )

    # Announce execution mode up-front so it's never ambiguous whether the
    # pipeline will run on Lambda or the caller's machine. Silent local-mode
    # fallback was confusing: the user would kick off run-project expecting
    # cloud execution and not realize their laptop was doing all the work.
    if local:
        logger.info(
            f"Mode: LOCAL (max_workers={max_workers}) — workers run in this process. "
            "Pass no --local to fan out via Lithops/Lambda."
        )
    else:
        lithops_env = os.environ.get("LITHOPS_ENV", "<unset>")
        logger.info(
            f"Mode: LITHOPS / AWS Lambda (LITHOPS_ENV={lithops_env}, "
            f"predict runtime={cfg.predict.runtime}). Pass --local to run in-process instead."
        )

    # ── Step 1: Extract project polygons ──
    if input_csv:
        logger.info(f"Step 1/7: Extracting polygons from request CSV '{input_csv}'")
        gdf, geojson_path, meta = _extract_from_request_csv(input_csv, geoparquet)
    elif has_filter:
        logger.info(f"Step 1/7: Extracting polygons from {geoparquet} via filter")
        gdf, geojson_path, meta = _extract_by_filter(
            geoparquet,
            where_sql=where_sql,
            poly_uuids=poly_uuids,
            cohorts=cohorts,
            project_ids=project_ids,
            short_names=short_names,
            framework_keys=framework_keys,
            year_override=year,
        )
    else:
        logger.info(f"Step 1/7: Extracting project '{short_name}' from {geoparquet}")
        gdf, geojson_path, meta = _extract_project(short_name, geoparquet, year_override=year)

    label = meta["label"]
    year_suffix = f"_{year}" if year is not None else ""
    if check_only:
        output = output or f"temp/{label}_missing_tiles{year_suffix}.csv"
    else:
        output = output or f"temp/{label}_stats{year_suffix}.csv"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    year_counts = meta["year_counts"]
    years_needed = sorted(year_counts.keys())

    # ── Step 2: Report TTC coverage ──
    logger.info(f"Step 2/7: TTC coverage report")
    logger.info(
        f"  Source: {label} ({meta['country']}, {meta['framework_key']})"
    )
    logger.info(f"  Polygons: {meta['n_polygons']}")
    if len(year_counts) == 1:
        logger.info(f"  Prediction year: {years_needed[0]}")
    else:
        logger.info(f"  Prediction years: {dict(year_counts)}")
    logger.info(
        f"  TTC status: {meta['n_with_ttc']} have TTC, "
        f"{meta['n_correct_yr']} have correct year, "
        f"{meta['n_missing_ttc']} missing"
    )

    # Filter to polygons missing TTC if requested
    if missing_only:
        ttc_col = "ttc" if "ttc" in gdf.columns else None
        if ttc_col:
            keep = []
            for idx, row in gdf.iterrows():
                ttc = row[ttc_col]
                pred_yr = row["pred_year"]
                has_correct = (
                    pd.notna(ttc)
                    and isinstance(ttc, (dict, list))
                    and len(ttc) > 0
                    and pred_yr in ttc
                )
                if not has_correct:
                    keep.append(idx)
            gdf = gdf.loc[keep].copy()
            logger.info(
                f"  --missing-only: filtered to {len(gdf)} polygons without correct-year TTC"
            )
            if gdf.empty:
                logger.info("All polygons already have TTC — nothing to do")
                return {
                    "label": label,
                    "n_polygons": meta["n_polygons"],
                    "n_tiles": 0,
                    "n_tiles_generated": 0,
                    "output_path": output,
                    "exit_code": ExitCode.NO_WORK,
                }
            # Update year counts after filtering
            year_counts = Counter(gdf["pred_year"])
            years_needed = sorted(year_counts.keys())

    # ── Step 3: Identify required tiles ──
    logger.info(f"Step 3/7: Identifying tiles across {len(years_needed)} year(s)")
    tiles = _identify_tiles(
        gdf,
        lookup_parquet=cfg.zonal.lookup_parquet or cfg.parquet_path,
        lookup_csv=cfg.zonal.lookup_csv,
    )

    if not tiles:
        logger.error("No tiles intersect the project polygons")
        return {
            "label": label,
            "n_polygons": meta["n_polygons"],
            "n_tiles": 0,
            "n_tiles_generated": 0,
            "output_path": output,
            "exit_code": ExitCode.NO_WORK,
        }

    # Summarise tiles per year
    tiles_by_year: dict[int, list] = {}
    for t in tiles:
        tiles_by_year.setdefault(t["year"], []).append(t)
    for yr in sorted(tiles_by_year):
        logger.info(f"  Year {yr}: {len(tiles_by_year[yr])} tiles")
    logger.info(f"  Total: {len(tiles)} unique tiles")

    # ── Step 4: Check for existing predictions ──
    logger.info(f"Step 4/7: Checking existing predictions at {dest}")
    from gri_tile_pipeline.tiles.availability import check_availability

    avail = check_availability(
        tiles, dest, check_type="predictions", region=cfg.zonal.tile_region,
    )
    existing = avail["existing"]
    missing = avail["missing"]
    logger.info(f"  Existing: {len(existing)}, Missing: {len(missing)}")

    if check_only:
        from gri_tile_pipeline.tiles.csv_io import write_tiles_csv

        if missing:
            write_tiles_csv(output, missing)
            logger.info(f"Wrote {len(missing)} missing tiles to {output}")
        else:
            logger.info("All prediction tiles exist — nothing to generate")
        return {
            "label": label,
            "n_polygons": meta["n_polygons"],
            "n_tiles": len(tiles),
            "n_tiles_missing": len(missing),
            "n_tiles_generated": 0,
            "output_path": output,
            "exit_code": ExitCode.SUCCESS,
        }

    if dry_run:
        logger.info("── Dry run ── would generate the following tiles:")
        for t in sorted(missing, key=lambda t: (t["year"], t["X_tile"], t["Y_tile"])):
            logger.info(f"  {t['year']}/{t['X_tile']}X{t['Y_tile']}Y")
        logger.info(f"Then run zonal stats for {meta['n_polygons']} polygons → {output}")
        return {
            "label": label,
            "n_polygons": meta["n_polygons"],
            "n_tiles": len(tiles),
            "n_tiles_missing": len(missing),
            "n_tiles_generated": 0,
            "output_path": output,
            "exit_code": ExitCode.SUCCESS,
        }

    n_generated = 0
    tiles_csv = None

    try:
        if missing:
            # Write missing tiles to temp CSV for download/predict steps
            from gri_tile_pipeline.tiles.csv_io import write_tiles_csv

            tiles_csv = os.path.join(
                os.path.dirname(str(geojson_path)), f"{label}_tiles.csv"
            )
            write_tiles_csv(tiles_csv, missing)

            # ── Step 5: Download ARD ──
            logger.info(f"Step 5/7: Downloading ARD for {len(missing)} tiles")
            if local:
                from gri_tile_pipeline.steps.download_ard import run_download_ard_local

                dl_tracker = run_download_ard_local(
                    tiles_csv, dest, cfg,
                    skip_existing=skip_existing,
                    max_workers=max_workers,
                )
            else:
                from gri_tile_pipeline.steps.download_ard import run_download_ard

                dl_tracker = run_download_ard(
                    tiles_csv, dest, cfg,
                    skip_existing=skip_existing,
                )
            dl_exit = exit_code_from_tracker(dl_tracker)
            if dl_exit == ExitCode.TOTAL_FAILURE:
                logger.error("All ARD downloads failed")
                return {
                    "label": label,
                    "n_polygons": meta["n_polygons"],
                    "n_tiles": len(tiles),
                    "n_tiles_generated": 0,
                    "output_path": output,
                    "exit_code": ExitCode.TOTAL_FAILURE,
                }

            # ── Step 6: Run inference ──
            logger.info(f"Step 6/7: Running inference for {len(missing)} tiles")

            # Verify ARD completeness
            ard_avail = check_availability(
                missing, dest, check_type="raw_ard", region=cfg.zonal.tile_region,
            )
            ready = ard_avail["existing"]
            not_ready = ard_avail["missing"]
            if not_ready:
                logger.warning(f"{len(not_ready)} tiles lack complete ARD — skipping")

            if ready:
                predict_csv = tiles_csv + ".predict.csv"
                write_tiles_csv(predict_csv, ready)

                if local:
                    from gri_tile_pipeline.steps.predict import run_predict_local

                    pred_tracker = run_predict_local(
                        predict_csv, dest, cfg,
                        skip_existing=skip_existing,
                        max_workers=max_workers,
                    )
                else:
                    from gri_tile_pipeline.steps.predict import run_predict

                    pred_tracker = run_predict(
                        predict_csv, dest, cfg,
                        skip_existing=skip_existing,
                    )
                pred_exit = exit_code_from_tracker(pred_tracker)
                if pred_exit == ExitCode.TOTAL_FAILURE:
                    logger.error("All prediction jobs failed")
                else:
                    n_generated = len(ready) - sum(
                        1 for r in pred_tracker.results
                        if r.status not in ("success", "partial")
                    )
            else:
                logger.warning("No tiles have complete ARD — skipping inference")
        else:
            logger.info("Step 5/7: Skipped (all prediction tiles exist)")
            logger.info("Step 6/7: Skipped (all prediction tiles exist)")

        # ── Step 7: Zonal stats (per-year groups) ──
        logger.info(f"Step 7/7: Running zonal stats → {output}")
        from gri_tile_pipeline.steps.zonal_stats import run_zonal_stats

        if lulc_raster:
            cfg.zonal.lulc_raster_path = lulc_raster

        tiles_bucket = _resolve_tiles_bucket(dest)
        tmp_dir = os.path.dirname(str(geojson_path))

        if len(years_needed) == 1:
            # Single year — straightforward
            run_zonal_stats(
                str(geojson_path), tiles_bucket, years_needed[0], output, cfg,
            )
        else:
            # Multiple years: split polygons by pred_year, run stats per group,
            # concatenate results.
            partial_csvs = []
            for yr in years_needed:
                year_gdf = gdf[gdf["pred_year"] == yr].copy()
                if year_gdf.empty:
                    continue

                year_geojson = os.path.join(tmp_dir, f"{label}_{yr}.geojson")
                year_gdf.to_file(year_geojson, driver="GeoJSON")
                year_csv = os.path.join(tmp_dir, f"{label}_{yr}_stats.csv")

                logger.info(
                    f"  Year {yr}: {len(year_gdf)} polygons"
                )
                run_zonal_stats(
                    year_geojson, tiles_bucket, yr, year_csv, cfg,
                )
                partial_csvs.append((yr, year_csv))

            # Concatenate partial results
            frames = []
            for yr, csv_path in partial_csvs:
                if os.path.exists(csv_path):
                    part = pd.read_csv(csv_path)
                    part["pred_year"] = yr
                    frames.append(part)
            if frames:
                combined = pd.concat(frames, ignore_index=True)
                combined.to_csv(output, index=False)
                logger.info(
                    f"Combined {len(frames)} year group(s) → "
                    f"{len(combined)} rows in {output}"
                )
            else:
                logger.error("No zonal stats results produced")

        logger.info(f"Done: {output}")
        return {
            "label": label,
            "n_polygons": meta["n_polygons"],
            "n_tiles": len(tiles),
            "n_tiles_generated": n_generated,
            "output_path": output,
            "exit_code": ExitCode.SUCCESS,
        }

    finally:
        # Cleanup temp CSVs
        for f in [tiles_csv, (tiles_csv + ".predict.csv") if tiles_csv else None]:
            if f:
                try:
                    os.remove(f)
                except OSError:
                    pass
