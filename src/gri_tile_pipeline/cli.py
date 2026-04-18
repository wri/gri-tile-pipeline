"""Click CLI: ``gri-ttc`` command group."""

from __future__ import annotations

import click
from loguru import logger

from gri_tile_pipeline.config import load_config
from gri_tile_pipeline.exit_codes import ExitCode, exit_code_from_tracker
from gri_tile_pipeline.logging import bind_run_context, new_run_id, setup_logging


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group(invoke_without_command=True)
@click.version_option(package_name="gri-tile-pipeline", prog_name="gri-ttc")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True),
              help="Path to pipeline YAML config.")
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False))
@click.option("--log-format", default="text",
              type=click.Choice(["text", "json"]),
              help="Log output format.")
@click.option("--run-id", default=None, help="Override auto-generated run ID.")
@click.option("--show-config", is_flag=True, help="Print resolved config as YAML and exit.")
@click.pass_context
def gri_ttc(ctx: click.Context, config_path, log_level, log_format, run_id, show_config):
    """GRI TTC tile pipeline."""
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = load_config(config_path)

    # Logging with run context
    run_id = run_id or new_run_id()
    ctx.obj["run_id"] = run_id
    bind_run_context(run_id)
    setup_logging(level=log_level, fmt=log_format)

    if show_config:
        import dataclasses
        import yaml as _yaml
        click.echo(_yaml.dump(dataclasses.asdict(ctx.obj["cfg"]), default_flow_style=False))
        ctx.exit(ExitCode.SUCCESS)
        return

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@gri_ttc.command()
@click.argument("input_json", type=click.Path(exists=True))
@click.option("--parquet", default=None, help="Path to tiledb.parquet.")
@click.option("-o", "--output", default="tiles.csv", help="Output tiles CSV path.")
@click.option("--x-col", default="X_tile")
@click.option("--y-col", default="Y_tile")
@click.option("--limit", type=int, default=None)
@click.pass_context
def ingest(ctx, input_json, parquet, output, x_col, y_col, limit):
    """Convert inbound JSON request to tiles CSV."""
    from gri_tile_pipeline.steps.ingest import run_ingest

    cfg = ctx.obj["cfg"]
    parquet = parquet or cfg.parquet_path

    n = run_ingest(input_json, parquet, output, x_col=x_col, y_col=y_col, limit=limit)
    if n == 0:
        ctx.exit(ExitCode.NO_WORK)
    else:
        ctx.exit(ExitCode.SUCCESS)


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

@gri_ttc.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--dest", required=True, help="S3 destination (s3://bucket/prefix).")
@click.option("--year", type=int, default=None,
              help="Prediction year (required for polygon file input).")
@click.option("--year-from-plantstart", is_flag=True,
              help="Derive prediction year from plantstart column (plantstart - 1).")
@click.option("--geoparquet", default=None,
              help="TerraMatch geoparquet path (for request CSV input).")
@click.option("--check-type", default="predictions",
              type=click.Choice(["raw_ard", "predictions"]),
              help="What to check for.")
@click.option("-o", "--output", default="missing.csv", help="Output CSV of missing tiles.")
@click.option("--save-polygons", default=None, type=click.Path(),
              help="Save resolved polygons to this path (GeoJSON).")
@click.option("--region", default=None, help="AWS region (default: from config tile_region).")
@click.option("--exit-on-missing", is_flag=True,
              help="Exit with code 10 (TILES_MISSING) when tiles are missing.")
@click.pass_context
def check(ctx, input_path, dest, year, year_from_plantstart, geoparquet,
          check_type, output, save_polygons, region, exit_on_missing):
    """Check tile availability on S3, output missing tiles.

    INPUT_PATH can be:

    \b
      - Tiles CSV (Year, X, Y, X_tile, Y_tile)  -> checks directly
      - Request CSV (project_id, plantstart_year) -> resolves via geoparquet
      - Polygon file (GeoJSON, GeoPackage, etc.)  -> spatial join, needs --year
      - JSON request (legacy inbound format)       -> resolves via tile lookup

    \b
    Examples:
        gri-ttc check tiles.csv --dest s3://tof-output --check-type raw_ard
        gri-ttc check request.csv --dest s3://tof-output --geoparquet temp/tm.geoparquet
        gri-ttc check polygons.geojson --dest s3://tof-output --year 2023
    """
    from gri_tile_pipeline.steps.resolve_input import resolve_to_tiles
    from gri_tile_pipeline.tiles.availability import check_availability
    from gri_tile_pipeline.tiles.csv_io import write_tiles_csv

    cfg = ctx.obj["cfg"]
    region = region or cfg.zonal.tile_region

    resolved = resolve_to_tiles(
        input_path, cfg,
        year=year,
        year_from_plantstart=year_from_plantstart,
        geoparquet=geoparquet,
    )
    tiles = resolved.tiles

    if not tiles:
        logger.warning("No tiles resolved from input")
        ctx.exit(ExitCode.NO_WORK)
        return

    if save_polygons and resolved.polygons_gdf is not None:
        resolved.polygons_gdf.to_file(save_polygons, driver="GeoJSON")
        logger.info(f"Polygons saved to {save_polygons}")

    result = check_availability(tiles, dest, check_type=check_type, region=region)
    missing = result["missing"]
    existing = result["existing"]
    logger.info(f"{len(existing)} existing, {len(missing)} missing out of {len(tiles)} tiles")

    if missing:
        write_tiles_csv(output, missing)
        logger.info(f"Missing tiles written to {output}")
        if exit_on_missing:
            ctx.exit(ExitCode.TILES_MISSING)
        else:
            ctx.exit(ExitCode.SUCCESS)
    else:
        logger.info("All tiles exist")
        ctx.exit(ExitCode.SUCCESS)


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

@gri_ttc.command()
@click.argument("tiles_csv", type=click.Path(exists=True))
@click.option("--dest", required=True, envvar="DEST",
              help="Output root (s3://bucket/prefix or local path).")
@click.option("--runtime", default=None)
@click.option("--mem", type=int, default=None, help="Lambda memory in MB.")
@click.option("--retries", type=int, default=None)
@click.option("--euc1-cfg", default=None)
@click.option("--usw2-cfg", default=None)
@click.option("--report-dir", default="job_reports")
@click.option("--skip-existing", is_flag=True, help="Skip tiles already on S3.")
@click.option("--local", is_flag=True, help="Run workers locally instead of via Lithops/Lambda.")
@click.option("--max-workers", type=int, default=1, help="Parallel workers for --local mode.")
@click.option("--debug", is_flag=True)
@click.option("--dry-run", is_flag=True, help="Preview what would happen without submitting jobs.")
@click.option("--yes", is_flag=True, help="Skip interactive approval prompt.")
@click.pass_context
def download(ctx, tiles_csv, dest, runtime, mem, retries, euc1_cfg, usw2_cfg,
             report_dir, skip_existing, local, max_workers, debug, dry_run, yes):
    """Fan out DEM + S1 RTC + S2 download jobs via Lithops (or locally with --local).

    S1 uses Planetary Computer RTC by default. For the legacy Earth Search
    GRD-based S1, use ``download-s1-legacy``.
    """
    from gri_tile_pipeline.steps.download_ard import estimate_cost
    from gri_tile_pipeline.tiles.csv_io import read_tiles_csv

    cfg = ctx.obj["cfg"]
    memory = mem or cfg.download.memory_mb

    # Cost estimate
    tiles = read_tiles_csv(tiles_csv)
    n = len(tiles)

    if not local:
        costs = estimate_cost(n, memory)
        logger.info(
            f"Cost estimate for {n} tiles @ {memory} MB: "
            f"DEM=${costs['DEM']:.2f}, S1=${costs['S1']:.2f}, S2=${costs['S2']:.2f}, "
            f"total=${costs['total']:.2f}"
        )

    if dry_run:
        if local:
            click.echo(f"Tiles: {n}, Mode: local (max_workers={max_workers})")
        else:
            click.echo(f"Tiles: {n}, Regions: eu-central-1 (DEM+S1), us-west-2 (S2)")
            click.echo(f"Invocations: {n * 3} (3 per tile)")
            click.echo(f"Estimated cost: ${costs['total']:.2f}")
        if n <= 5:
            for t in tiles:
                click.echo(f"  {t['X_tile']}X{t['Y_tile']}Y")
        else:
            for t in tiles[:3]:
                click.echo(f"  {t['X_tile']}X{t['Y_tile']}Y")
            click.echo(f"  ... and {n - 3} more")
        ctx.exit(ExitCode.SUCCESS)
        return

    if not yes:
        click.confirm("Proceed with job submission?", abort=True)

    if local:
        from gri_tile_pipeline.steps.download_ard import run_download_ard_local
        tracker = run_download_ard_local(
            tiles_csv, dest, cfg,
            report_dir=report_dir, debug=debug,
            skip_existing=skip_existing, max_workers=max_workers,
        )
    else:
        from gri_tile_pipeline.steps.download_ard import run_download_ard
        tracker = run_download_ard(
            tiles_csv, dest, cfg,
            runtime=runtime, memory_mb=mem, retries=retries,
            euc1_cfg=euc1_cfg, usw2_cfg=usw2_cfg,
            report_dir=report_dir, debug=debug,
            skip_existing=skip_existing,
        )

    ctx.exit(exit_code_from_tracker(tracker))


# ---------------------------------------------------------------------------
# download-s1-legacy
# ---------------------------------------------------------------------------

@gri_ttc.command("download-s1-legacy")
@click.argument("tiles_csv", type=click.Path(exists=True))
@click.option("--dest", required=True, envvar="DEST")
@click.option("--runtime", default=None)
@click.option("--mem", type=int, default=None)
@click.option("--retries", type=int, default=None)
@click.option("--s1-cfg", default=None)
@click.option("--pc-token-cache", default=".pc_sas_token_cache.json")
@click.option("--pc-token-min-ttl-minutes", type=int, default=20)
@click.option("--report-dir", default="job_reports")
@click.option("--skip-existing", is_flag=True)
@click.option("--local", is_flag=True, help="Run workers locally instead of via Lithops/Lambda.")
@click.option("--max-workers", type=int, default=1, help="Parallel workers for --local mode.")
@click.option("--debug", is_flag=True)
@click.option("--dry-run", is_flag=True, help="Preview what would happen without submitting jobs.")
@click.option("--yes", is_flag=True, help="Skip interactive approval prompt.")
@click.pass_context
def download_s1(ctx, tiles_csv, dest, runtime, mem, retries, s1_cfg,
                pc_token_cache, pc_token_min_ttl_minutes,
                report_dir, skip_existing, local, max_workers, debug, dry_run, yes):
    """Fan out S1 RTC acquisition jobs via Lithops (or locally with --local).

    This is a standalone S1-only command. The default ``download`` command
    already includes S1 RTC — use this only if you need to run S1 separately.
    """
    from gri_tile_pipeline.tiles.csv_io import read_tiles_csv

    cfg = ctx.obj["cfg"]

    tiles = read_tiles_csv(tiles_csv)
    n = len(tiles)
    logger.info(f"S1 RTC download: {n} tiles")

    if dry_run:
        if local:
            click.echo(f"Tiles: {n}, Mode: local (max_workers={max_workers})")
        else:
            click.echo(f"Tiles: {n}, Region: us-west-2 (S1 RTC from Planetary Computer)")
            click.echo(f"Invocations: {n}")
        if n <= 5:
            for t in tiles:
                click.echo(f"  {t['X_tile']}X{t['Y_tile']}Y")
        else:
            for t in tiles[:3]:
                click.echo(f"  {t['X_tile']}X{t['Y_tile']}Y")
            click.echo(f"  ... and {n - 3} more")
        ctx.exit(ExitCode.SUCCESS)
        return

    if not yes:
        click.confirm("Proceed with S1 RTC job submission?", abort=True)

    if local:
        from gri_tile_pipeline.steps.download_s1_rtc import run_download_s1_rtc_local
        tracker = run_download_s1_rtc_local(
            tiles_csv, dest, cfg,
            pc_token_cache=pc_token_cache,
            pc_token_min_ttl_minutes=pc_token_min_ttl_minutes,
            report_dir=report_dir, debug=debug,
            skip_existing=skip_existing, max_workers=max_workers,
        )
    else:
        from gri_tile_pipeline.steps.download_s1_rtc import run_download_s1_rtc
        tracker = run_download_s1_rtc(
            tiles_csv, dest, cfg,
            runtime=runtime, memory_mb=mem, retries=retries,
            s1_cfg=s1_cfg,
            pc_token_cache=pc_token_cache,
            pc_token_min_ttl_minutes=pc_token_min_ttl_minutes,
            report_dir=report_dir, debug=debug,
            skip_existing=skip_existing,
        )

    ctx.exit(exit_code_from_tracker(tracker))


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

@gri_ttc.command()
@click.argument("tiles_csv", type=click.Path(exists=True))
@click.option("--dest", required=True, envvar="DEST")
@click.option("--model-path", default=None, help="S3 or local path to model files.")
@click.option("--mem", type=int, default=None)
@click.option("--retries", type=int, default=None)
@click.option("--skip-existing", is_flag=True)
@click.option("--report-dir", default="job_reports")
@click.option("--local", is_flag=True, help="Run workers locally instead of via Lithops/Lambda.")
@click.option("--max-workers", type=int, default=1, help="Parallel workers for --local mode.")
@click.option("--debug", is_flag=True)
@click.option("--dry-run", is_flag=True, help="Preview what would happen without submitting jobs.")
@click.option("--aws-profile", default=None, help="AWS profile name (for --local S3 access).")
@click.option("--yes", is_flag=True)
@click.pass_context
def predict(ctx, tiles_csv, dest, model_path, mem, retries,
            skip_existing, report_dir, local, max_workers, debug, dry_run, aws_profile, yes):
    """Fan out tree cover prediction jobs via Lithops (or locally with --local)."""
    from gri_tile_pipeline.steps.predict import AVG_PREDICT_DURATION
    from gri_tile_pipeline.steps.download_ard import PRICE_PER_GB_SEC
    from gri_tile_pipeline.tiles.csv_io import read_tiles_csv

    cfg = ctx.obj["cfg"]

    if dry_run:
        tiles = read_tiles_csv(tiles_csv)
        n = len(tiles)
        if local:
            click.echo(f"Tiles: {n}, Mode: local (max_workers={max_workers})")
        else:
            pred_mem = (mem or cfg.predict.memory_mb)
            pred_cost = n * AVG_PREDICT_DURATION * (pred_mem / 1024.0) * PRICE_PER_GB_SEC
            click.echo(f"Tiles: {n}, Region: us-west-2, Memory: {pred_mem} MB")
            click.echo(f"Invocations: {n}")
            click.echo(f"Estimated cost: ${pred_cost:.2f}")
        if n <= 5:
            for t in tiles:
                click.echo(f"  {t['X_tile']}X{t['Y_tile']}Y")
        else:
            for t in tiles[:3]:
                click.echo(f"  {t['X_tile']}X{t['Y_tile']}Y")
            click.echo(f"  ... and {n - 3} more")
        ctx.exit(ExitCode.SUCCESS)
        return

    if not yes:
        click.confirm("Proceed with prediction job submission?", abort=True)

    model_path = model_path or cfg.predict.model_path

    if local:
        from gri_tile_pipeline.steps.predict import run_predict_local
        tracker = run_predict_local(
            tiles_csv, dest, cfg,
            model_path=model_path,
            skip_existing=skip_existing,
            report_dir=report_dir, debug=debug,
            max_workers=max_workers,
            aws_profile=aws_profile,
        )
    else:
        from gri_tile_pipeline.steps.predict import run_predict
        tracker = run_predict(
            tiles_csv, dest, cfg,
            model_path=model_path,
            memory_mb=mem, retries=retries,
            skip_existing=skip_existing,
            report_dir=report_dir, debug=debug,
        )

    ctx.exit(exit_code_from_tracker(tracker))


# ---------------------------------------------------------------------------
# stats (stub — Phase 4)
# ---------------------------------------------------------------------------

@gri_ttc.command()
@click.argument("polygons", type=click.Path(exists=True))
@click.option("--tiles-bucket", default=None, help="S3 bucket with prediction tiles.")
@click.option("--year", type=int, required=True, help="Year to fetch predictions for.")
@click.option("-o", "--output", default="results.csv", help="Output results CSV path.")
@click.option("--lookup-parquet", default=None, type=click.Path(),
              help="Tile lookup parquet (overrides config).")
@click.option("--lookup-csv", default=None, type=click.Path(),
              help="Tile lookup CSV (fallback if parquet unavailable).")
@click.option("--include-cols", default=None, type=str,
              help="Comma-separated polygon columns to include in output (auto-detects if omitted).")
@click.option("--lulc-raster", default=None, type=str,
              help="Path or S3 URI to LULC raster (overrides config zonal.lulc_raster_path).")
@click.option("--shift-error/--no-shift-error", default=None,
              help="Enable/disable shift error calculation (overrides config).")
@click.pass_context
def stats(ctx, polygons, tiles_bucket, year, output, lookup_parquet, lookup_csv, include_cols, lulc_raster, shift_error):
    """Calculate zonal tree cover statistics for polygons."""
    from gri_tile_pipeline.steps.zonal_stats import run_zonal_stats

    cfg = ctx.obj["cfg"]
    bucket = tiles_bucket or cfg.zonal.tile_bucket
    cols = [c.strip() for c in include_cols.split(",")] if include_cols else None

    if lulc_raster:
        cfg.zonal.lulc_raster_path = lulc_raster
    if shift_error is not None:
        cfg.zonal.shift_error_enabled = shift_error

    run_zonal_stats(
        polygons, bucket, year, output, cfg,
        lookup_parquet=lookup_parquet,
        lookup_csv=lookup_csv,
        include_cols=cols,
    )
    ctx.exit(ExitCode.SUCCESS)


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------

@gri_ttc.command()
@click.argument("tiles_csv", type=click.Path(exists=True))
@click.option("--mem", type=int, default=None, help="Lambda memory in MB.")
@click.option("--include-predict", is_flag=True, help="Include prediction cost estimate.")
@click.pass_context
def cost(ctx, tiles_csv, mem, include_predict):
    """Estimate Lambda costs without executing."""
    from gri_tile_pipeline.steps.download_ard import estimate_cost, AVG_DURATIONS, PRICE_PER_GB_SEC
    from gri_tile_pipeline.steps.predict import AVG_PREDICT_DURATION
    from gri_tile_pipeline.tiles.csv_io import read_tiles_csv

    cfg = ctx.obj["cfg"]
    memory = mem or cfg.download.memory_mb
    tiles = read_tiles_csv(tiles_csv)
    n = len(tiles)

    costs = estimate_cost(n, memory)
    click.echo(f"Tiles: {n}, Memory: {memory} MB")
    for task_type, avg_sec in AVG_DURATIONS.items():
        click.echo(f"  {task_type:6s} ${costs[task_type]:.2f}  (avg {avg_sec}s x {n} jobs)")
    click.echo(f"  Total: ${costs['total']:.2f}")

    if include_predict:
        pred_mem = cfg.predict.memory_mb
        pred_mem_gb = pred_mem / 1024.0
        pred_cost = n * AVG_PREDICT_DURATION * pred_mem_gb * PRICE_PER_GB_SEC
        click.echo(f"\nPrediction ({pred_mem} MB, ~{AVG_PREDICT_DURATION}s avg):")
        click.echo(f"  Predict: ${pred_cost:.2f}")
        click.echo(f"  Grand total: ${costs['total'] + pred_cost:.2f}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

VALID_STEPS = {"download", "predict", "stats"}


@gri_ttc.command()
@click.argument("tiles_csv", type=click.Path(exists=True))
@click.option("--dest", required=True, envvar="DEST",
              help="Output root (s3://bucket/prefix or local path).")
@click.option("--steps", default="download,predict",
              help="Comma-separated steps to run: download, predict, stats.")
@click.option("--polygons", default=None, type=click.Path(exists=True),
              help="Polygon file for zonal stats (required if 'stats' in --steps).")
@click.option("--year", type=int, default=None,
              help="Year for zonal stats (required if 'stats' in --steps).")
@click.option("-o", "--output", default="results.csv",
              help="Output CSV path for stats results.")
@click.option("--local", is_flag=True, help="Run workers locally instead of via Lithops/Lambda.")
@click.option("--max-workers", type=int, default=1, help="Parallel workers for --local mode.")
@click.option("--skip-existing", is_flag=True, help="Skip tiles already on S3.")
@click.option("--dry-run", is_flag=True, help="Preview what would happen without executing.")
@click.option("--yes", is_flag=True, help="Skip interactive approval prompt.")
@click.pass_context
def run(ctx, tiles_csv, dest, steps, polygons, year, output,
        local, max_workers, skip_existing, dry_run, yes):
    """Execute pipeline steps on a tiles CSV.

    Takes the output of ``gri-ttc check`` and runs the requested steps.
    Default steps: download, predict.

    \b
    Examples:
        gri-ttc run missing.csv --dest s3://tof-output --yes
        gri-ttc run missing.csv --dest s3://tof-output --local --max-workers 4
        gri-ttc run missing.csv --dest s3://tof-output --steps download,predict,stats \\
            --polygons polygons.geojson --year 2023
    """
    from gri_tile_pipeline.tiles.csv_io import read_tiles_csv, write_tiles_csv

    cfg = ctx.obj["cfg"]
    step_list = [s.strip() for s in steps.split(",")]

    unknown = set(step_list) - VALID_STEPS
    if unknown:
        raise click.UsageError(f"Unknown steps: {sorted(unknown)}. Valid: {sorted(VALID_STEPS)}")

    if "stats" in step_list:
        if not polygons:
            raise click.UsageError("--polygons is required when 'stats' is in --steps")
        if year is None:
            raise click.UsageError("--year is required when 'stats' is in --steps")

    tiles = read_tiles_csv(tiles_csv)
    n = len(tiles)
    if n == 0:
        logger.warning("No tiles in input CSV")
        ctx.exit(ExitCode.NO_WORK)
        return

    logger.info(f"Pipeline: {' -> '.join(step_list)} for {n} tiles")

    if dry_run:
        click.echo(f"Tiles: {n}, Steps: {' -> '.join(step_list)}")
        click.echo(f"Mode: {'local' if local else 'Lithops/Lambda'}")
        if n <= 5:
            for t in tiles:
                click.echo(f"  {t['X_tile']}X{t['Y_tile']}Y (year={t['year']})")
        else:
            for t in tiles[:3]:
                click.echo(f"  {t['X_tile']}X{t['Y_tile']}Y (year={t['year']})")
            click.echo(f"  ... and {n - 3} more")
        ctx.exit(ExitCode.SUCCESS)
        return

    if not yes:
        click.confirm(
            f"Run {' -> '.join(step_list)} for {n} tiles?", abort=True,
        )

    dl_exit = ExitCode.SUCCESS
    pred_exit = ExitCode.SUCCESS

    # -- Download --
    if "download" in step_list:
        logger.info(f"Step: download ({n} tiles)")
        if local:
            from gri_tile_pipeline.steps.download_ard import run_download_ard_local
            tracker = run_download_ard_local(
                tiles_csv, dest, cfg,
                skip_existing=skip_existing, max_workers=max_workers,
            )
        else:
            from gri_tile_pipeline.steps.download_ard import run_download_ard
            tracker = run_download_ard(tiles_csv, dest, cfg, skip_existing=skip_existing)
        dl_exit = exit_code_from_tracker(tracker)
        if dl_exit == ExitCode.TOTAL_FAILURE:
            logger.error("All download jobs failed")
            ctx.exit(ExitCode.TOTAL_FAILURE)
            return

    # -- Predict --
    if "predict" in step_list:
        # Verify ARD completeness before predicting
        from gri_tile_pipeline.tiles.availability import check_availability
        avail = check_availability(tiles, dest, check_type="raw_ard")
        ready = avail["existing"]
        not_ready = avail["missing"]

        if not_ready:
            logger.warning(f"{len(not_ready)} tiles lack complete ARD — skipping predict for them")
        if not ready:
            logger.error("No tiles have complete ARD — cannot predict")
            ctx.exit(ExitCode.TOTAL_FAILURE)
            return

        predict_csv = tiles_csv + ".predict.csv"
        write_tiles_csv(predict_csv, ready)
        logger.info(f"Step: predict ({len(ready)}/{n} tiles with ARD)")

        try:
            if local:
                from gri_tile_pipeline.steps.predict import run_predict_local
                pred_tracker = run_predict_local(
                    predict_csv, dest, cfg,
                    skip_existing=skip_existing, max_workers=max_workers,
                )
            else:
                from gri_tile_pipeline.steps.predict import run_predict
                pred_tracker = run_predict(predict_csv, dest, cfg, skip_existing=skip_existing)
            pred_exit = exit_code_from_tracker(pred_tracker)
            if pred_exit == ExitCode.TOTAL_FAILURE:
                logger.error("All prediction jobs failed")
                ctx.exit(ExitCode.TOTAL_FAILURE)
                return
        finally:
            import os as _os
            try:
                _os.remove(predict_csv)
            except OSError:
                pass

    # -- Stats --
    if "stats" in step_list:
        logger.info(f"Step: stats ({polygons}, year={year})")
        from gri_tile_pipeline.steps.zonal_stats import run_zonal_stats

        tiles_bucket = dest.replace("s3://", "").split("/")[0] if dest.startswith("s3://") else dest
        run_zonal_stats(polygons, tiles_bucket, year, output, cfg)

    any_failed = dl_exit != ExitCode.SUCCESS or pred_exit != ExitCode.SUCCESS
    ctx.exit(ExitCode.PARTIAL_FAILURE if any_failed else ExitCode.SUCCESS)


# ---------------------------------------------------------------------------
# run-project
# ---------------------------------------------------------------------------

@gri_ttc.command("run-project")
@click.argument("short_name", required=False, default=None)
@click.option("--input", "input_csv", default=None, type=click.Path(exists=True),
              help="Request CSV with project_id,plantstart_year columns.")
@click.option("--dest", required=True, envvar="DEST",
              help="S3 URI (s3://bucket) or local path for ARD + predictions.")
@click.option("--geoparquet", default="temp/tm.geoparquet",
              help="Path to TerraMatch geoparquet file.")
@click.option("--year", type=int, default=None,
              help="Override prediction year (default: plantstart - 1).")
@click.option("-o", "--output", default=None,
              help="Output CSV path.")
@click.option("--max-workers", type=int, default=1,
              help="Parallel workers for local download/predict.")
@click.option("--skip-existing/--no-skip-existing", default=True,
              help="Skip tiles already available at dest.")
@click.option("--lulc-raster", default=None,
              help="Path or S3 URI to LULC raster for error propagation.")
@click.option("--shift-error/--no-shift-error", default=None,
              help="Enable/disable shift error calculation.")
@click.option("--missing-only", is_flag=True,
              help="Only process polygons missing TTC for their prediction year.")
@click.option("--check-only", is_flag=True,
              help="Stop after step 4: write missing tiles CSV and exit.")
@click.option("--dry-run", is_flag=True,
              help="Show what would happen without executing.")
@click.option("--yes", is_flag=True, help="Skip confirmation prompts.")
@click.pass_context
def run_project(ctx, short_name, input_csv, dest, geoparquet, year, output,
                max_workers, skip_existing, lulc_raster, shift_error,
                missing_only, check_only, dry_run, yes):
    """End-to-end pipeline for a TerraMatch project.

    Provide either a SHORT_NAME argument or --input with a request CSV
    containing (project_id, plantstart_year) pairs.

    \b
    Examples:
        gri-ttc run-project GHA_22_INEC --dest s3://tof-output --yes
        gri-ttc run-project --input request.csv --dest s3://tof-output --dry-run
        gri-ttc run-project GHA_22_INEC --dest ./local_tiles --dry-run
    """
    from gri_tile_pipeline.steps.project_e2e import run_project_pipeline

    if not short_name and not input_csv:
        raise click.UsageError("Provide either SHORT_NAME or --input CSV.")
    if short_name and input_csv:
        raise click.UsageError("Provide SHORT_NAME or --input, not both.")

    cfg = ctx.obj["cfg"]

    if shift_error is not None:
        cfg.zonal.shift_error_enabled = shift_error

    if not dry_run and not check_only and not yes:
        source = short_name or input_csv
        click.confirm(
            f"Run full pipeline for '{source}'? "
            f"(dest={dest}, geoparquet={geoparquet})",
            abort=True,
        )

    try:
        result = run_project_pipeline(
            short_name,
            dest,
            cfg,
            input_csv=input_csv,
            geoparquet=geoparquet,
            year=year,
            output=output,
            max_workers=max_workers,
            skip_existing=skip_existing,
            lulc_raster=lulc_raster,
            missing_only=missing_only,
            check_only=check_only,
            dry_run=dry_run,
        )
    except ValueError as e:
        logger.error(str(e))
        ctx.exit(ExitCode.BAD_INPUT)
        return

    ctx.exit(result["exit_code"])
