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
@click.argument("tiles_csv", type=click.Path(exists=True))
@click.option("--dest", required=True, help="S3 destination (s3://bucket/prefix).")
@click.option("--check-type", default="raw_ard",
              type=click.Choice(["raw_ard", "predictions"]),
              help="What to check for.")
@click.option("-o", "--output", default="missing.csv", help="Output CSV of missing tiles.")
@click.option("--exit-on-missing", is_flag=True,
              help="Exit with code 10 (TILES_MISSING) when tiles are missing.")
@click.pass_context
def check(ctx, tiles_csv, dest, check_type, output, exit_on_missing):
    """Check tile availability on S3, output missing tiles."""
    from gri_tile_pipeline.tiles.availability import check_availability
    from gri_tile_pipeline.tiles.csv_io import read_tiles_csv, write_tiles_csv

    tiles = read_tiles_csv(tiles_csv)
    result = check_availability(tiles, dest, check_type=check_type)

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
    """Fan out DEM + S1 + S2 download jobs via Lithops (or locally with --local)."""
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
# download-s1
# ---------------------------------------------------------------------------

@gri_ttc.command("download-s1")
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
    """Fan out S1 RTC acquisition jobs via Lithops (or locally with --local)."""
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
@click.option("--yes", is_flag=True)
@click.pass_context
def predict(ctx, tiles_csv, dest, model_path, mem, retries,
            skip_existing, report_dir, local, max_workers, debug, dry_run, yes):
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

    if local:
        from gri_tile_pipeline.steps.predict import run_predict_local
        tracker = run_predict_local(
            tiles_csv, dest, cfg,
            model_path=model_path,
            skip_existing=skip_existing,
            report_dir=report_dir, debug=debug,
            max_workers=max_workers,
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
@click.option("--tiles-bucket", default=None)
@click.option("--year", type=int, required=True)
@click.option("-o", "--output", default="results.csv")
@click.pass_context
def stats(ctx, polygons, tiles_bucket, year, output):
    """Calculate zonal tree cover statistics for polygons."""
    from gri_tile_pipeline.steps.zonal_stats import run_zonal_stats

    cfg = ctx.obj["cfg"]
    bucket = tiles_bucket or cfg.zonal.tile_bucket

    run_zonal_stats(polygons, bucket, year, output, cfg)
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

@gri_ttc.command()
@click.argument("input_json", type=click.Path(exists=True))
@click.option("--dest", required=True, envvar="DEST")
@click.option("--polygons", default=None, type=click.Path(exists=True),
              help="Polygon file for zonal stats (skip stats if omitted).")
@click.option("--parquet", default=None)
@click.option("--year", type=int, default=None)
@click.option("-o", "--output", default="results.csv")
@click.option("--tiles-csv", default=None, type=click.Path(),
              help="Explicit path for intermediate tiles CSV. Uses a tempfile by default.")
@click.option("--skip-existing", is_flag=True)
@click.option("--local", is_flag=True, help="Run workers locally instead of via Lithops/Lambda.")
@click.option("--max-workers", type=int, default=1, help="Parallel workers for --local mode.")
@click.option("--resume", "resume_run_id", default=None,
              help="Resume a previous run by its run ID.")
@click.option("--yes", is_flag=True)
@click.pass_context
def run(ctx, input_json, dest, polygons, parquet, year, output, tiles_csv,
        skip_existing, local, max_workers, resume_run_id, yes):
    """Execute full pipeline: ingest -> check -> download -> predict -> stats."""
    import dataclasses as _dc
    import tempfile as _tempfile
    from datetime import datetime, timezone

    from gri_tile_pipeline.steps.ingest import run_ingest
    from gri_tile_pipeline.tracking import PipelineRun, RunStore, StepResult, get_per_tile_status
    from gri_tile_pipeline.tiles.csv_io import read_tiles_csv

    cfg = ctx.obj["cfg"]
    run_id = ctx.obj["run_id"]
    parquet = parquet or cfg.parquet_path
    store = RunStore()

    # --resume: reload prior run state
    if resume_run_id:
        try:
            pipeline_run = store.load(resume_run_id)
            run_id = pipeline_run.run_id
            logger.info(f"Resuming run {run_id} (status={pipeline_run.status})")
        except FileNotFoundError:
            logger.error(f"Run {resume_run_id} not found in {store.base_dir}")
            ctx.exit(ExitCode.BAD_INPUT)
            return
    else:
        pipeline_run = PipelineRun(
            run_id=run_id,
            started_at=datetime.now(timezone.utc).isoformat(),
            config_snapshot=_dc.asdict(cfg),
            dest=dest,
        )

    # Use a tempfile unless the user explicitly specifies --tiles-csv
    tmp_file = None
    if tiles_csv is None:
        tmp_file = _tempfile.NamedTemporaryFile(suffix=".csv", prefix="gri_tiles_", delete=False)
        tiles_csv = tmp_file.name
        tmp_file.close()

    try:
        # 1. Ingest
        logger.info("Step 1/5: Ingest")
        pipeline_run.current_step = "ingest"
        n = run_ingest(input_json, parquet, tiles_csv)
        if n == 0:
            logger.warning("No tiles to process")
            pipeline_run.status = "completed"
            pipeline_run.finished_at = datetime.now(timezone.utc).isoformat()
            store.save(pipeline_run)
            ctx.exit(ExitCode.NO_WORK)
            return

        # Register tiles in run metadata
        tiles = read_tiles_csv(tiles_csv)
        for t in tiles:
            pipeline_run.ensure_tile(
                t["X_tile"], t["Y_tile"], t["year"], t["lon"], t["lat"],
            )
        store.save(pipeline_run)

        # 2. Download ARD
        logger.info("Step 2/5: Download ARD")
        pipeline_run.current_step = "download_ard"
        if not yes:
            click.confirm(f"Download {n} tiles?", abort=True)

        if local:
            from gri_tile_pipeline.steps.download_ard import run_download_ard_local
            tracker = run_download_ard_local(
                tiles_csv, dest, cfg, report_dir="job_reports",
                skip_existing=skip_existing, max_workers=max_workers,
            )
        else:
            from gri_tile_pipeline.steps.download_ard import run_download_ard
            tracker = run_download_ard(tiles_csv, dest, cfg, skip_existing=skip_existing)

        dl_exit = exit_code_from_tracker(tracker)
        # Record per-tile status
        for tile_key, status in get_per_tile_status(tracker).items():
            ts = pipeline_run.tiles.get(tile_key)
            if ts:
                ts.steps["download_ard"] = StepResult(status=status)
        store.save(pipeline_run)

        if dl_exit == ExitCode.TOTAL_FAILURE:
            logger.error("All download jobs failed")
            pipeline_run.status = "failed"
            pipeline_run.finished_at = datetime.now(timezone.utc).isoformat()
            store.save(pipeline_run)
            ctx.exit(ExitCode.TOTAL_FAILURE)
            return

        # 3. Predict — only tiles with complete ARD
        logger.info("Step 3/5: Predict")
        pipeline_run.current_step = "predict"

        # Filter to tiles that actually have all ARD available
        from gri_tile_pipeline.tiles.availability import check_availability
        avail = check_availability(tiles, dest, check_type="raw_ard")
        ready_tiles = avail["existing"]
        not_ready = avail["missing"]
        if not_ready:
            logger.warning(
                f"{len(not_ready)} tiles lack complete ARD — skipping predict for them"
            )
            for t in not_ready:
                key = pipeline_run.tile_key(t["X_tile"], t["Y_tile"])
                ts = pipeline_run.tiles.get(key)
                if ts:
                    ts.steps["predict"] = StepResult(status="skipped")

        if not ready_tiles:
            logger.error("No tiles have complete ARD — skipping predict")
            pipeline_run.status = "failed"
            pipeline_run.finished_at = datetime.now(timezone.utc).isoformat()
            store.save(pipeline_run)
            ctx.exit(ExitCode.TOTAL_FAILURE)
            return

        # Write filtered tiles CSV for predict step
        from gri_tile_pipeline.tiles.csv_io import write_tiles_csv
        predict_csv = tiles_csv + ".predict.csv"
        write_tiles_csv(predict_csv, ready_tiles)
        logger.info(f"{len(ready_tiles)}/{len(tiles)} tiles ready for prediction")

        if local:
            from gri_tile_pipeline.steps.predict import run_predict_local
            pred_tracker = run_predict_local(
                predict_csv, dest, cfg, skip_existing=skip_existing,
                max_workers=max_workers,
            )
        else:
            from gri_tile_pipeline.steps.predict import run_predict
            pred_tracker = run_predict(predict_csv, dest, cfg, skip_existing=skip_existing)

        pred_exit = exit_code_from_tracker(pred_tracker)
        for tile_key, status in get_per_tile_status(pred_tracker).items():
            ts = pipeline_run.tiles.get(tile_key)
            if ts:
                ts.steps["predict"] = StepResult(status=status)
        store.save(pipeline_run)

        if pred_exit == ExitCode.TOTAL_FAILURE:
            logger.error("All prediction jobs failed")
            pipeline_run.status = "failed"
            pipeline_run.finished_at = datetime.now(timezone.utc).isoformat()
            store.save(pipeline_run)
            ctx.exit(ExitCode.TOTAL_FAILURE)
            return

        # 4. Zonal stats (optional)
        if polygons:
            logger.info("Step 4/5: Zonal stats")
            pipeline_run.current_step = "zonal_stats"
            from gri_tile_pipeline.steps.zonal_stats import run_zonal_stats
            run_zonal_stats(polygons, cfg.zonal.tile_bucket, year, output, cfg)
        else:
            logger.info("Step 4/5: Skipped (no polygons provided)")

        logger.info("Step 5/5: Done")
        any_failed = dl_exit != ExitCode.SUCCESS or pred_exit != ExitCode.SUCCESS
        pipeline_run.status = "partial" if any_failed else "completed"
        pipeline_run.current_step = None
        pipeline_run.finished_at = datetime.now(timezone.utc).isoformat()
        store.save(pipeline_run)
        logger.info(f"Run metadata saved: .gri_runs/{run_id}.json")
        ctx.exit(ExitCode.PARTIAL_FAILURE if any_failed else ExitCode.SUCCESS)
    finally:
        import os as _os
        if tmp_file is not None:
            try:
                _os.remove(tiles_csv)
            except OSError:
                pass
        # Clean up predict-filtered CSV
        try:
            _os.remove(tiles_csv + ".predict.csv")
        except OSError:
            pass
