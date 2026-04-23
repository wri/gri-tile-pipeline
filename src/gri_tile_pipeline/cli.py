"""Click CLI: ``gri-ttc`` command group."""

from __future__ import annotations

import click
from loguru import logger

from gri_tile_pipeline.cli_context import (
    CliContext,
    attach,
    resolve_git_sha,
    resolve_pipeline_version,
)
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
# Legacy log controls — kept working; prefer -v/-q/--json for new usage.
@click.option("--log-level", default=None,
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
              help="Set log level explicitly (overrides -v/-q).")
@click.option("--log-format", default="text",
              type=click.Choice(["text", "json"]),
              help="Log output format.")
# Cross-cutting flags that every subcommand should honor.
@click.option("-v", "--verbose", count=True,
              help="Increase verbosity (-v=DEBUG, -vv=TRACE).")
@click.option("-q", "--quiet", is_flag=True,
              help="Suppress info logs (WARNING and above only).")
@click.option("--json", "json_mode", is_flag=True,
              help="Emit machine-readable JSON on stdout; text logs go to stderr.")
@click.option("--workers", type=int, default=1, show_default=True,
              help="Parallel workers for local-mode operations.")
@click.option("--dry-run", is_flag=True,
              help="Preview what would happen without executing side effects.")
@click.option("--yes", is_flag=True,
              help="Skip confirmation prompts.")
@click.option("--aws-profile", default=None,
              help="AWS credentials profile (overrides AWS_PROFILE env).")
@click.option("--run-history-dir", default="runs", show_default=True,
              type=click.Path(),
              help="Directory for JobTracker run reports.")
@click.option("--run-id", default=None, help="Override auto-generated run ID.")
@click.option("--show-config", is_flag=True, help="Print resolved config as YAML and exit.")
@click.pass_context
def gri_ttc(ctx: click.Context, config_path, log_level, log_format, verbose, quiet,
            json_mode, workers, dry_run, yes, aws_profile, run_history_dir,
            run_id, show_config):
    """GRI TTC tile pipeline."""
    if verbose and quiet:
        raise click.UsageError("-v and -q are mutually exclusive.")

    cfg = load_config(config_path)
    run_id = run_id or new_run_id()
    bind_run_context(run_id)
    setup_logging(level=log_level, fmt=log_format, verbose=verbose,
                  quiet=quiet, json_mode=json_mode)

    gri = CliContext(
        cfg=cfg,
        run_id=run_id,
        verbose=verbose,
        quiet=quiet,
        json_mode=json_mode,
        workers=workers,
        dry_run=dry_run,
        yes=yes,
        aws_profile=aws_profile,
        run_history_dir=run_history_dir,
        pipeline_version=resolve_pipeline_version(),
        git_sha=resolve_git_sha(),
    )
    attach(ctx, gri)

    if show_config:
        import dataclasses
        import yaml as _yaml
        click.echo(_yaml.dump(dataclasses.asdict(cfg), default_flow_style=False))
        ctx.exit(ExitCode.SUCCESS)
        return

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# resolve  (replaces legacy `ingest`)
# ---------------------------------------------------------------------------

@gri_ttc.command()
@click.argument("input", required=False, default=None)
@click.option("-o", "--output", default="tiles.csv",
              type=click.Path(), help="Output tiles CSV path.")
@click.option("--year", type=int, default=None,
              help="Prediction year (required for polygon file; optional for short_name/filter).")
@click.option("--year-from-plantstart", is_flag=True,
              help="Derive prediction year from plantstart column (polygon input).")
@click.option("--geoparquet", default=None,
              help="TerraMatch geoparquet path (for short_name, request CSV, or filter input).")
@click.option("--save-polygons", default=None, type=click.Path(),
              help="Save resolved polygons to this path (GeoJSON).")
@click.option("--project-id", "project_ids", multiple=True,
              help="Filter by project_id (repeatable; mutually exclusive with INPUT).")
@click.option("--short-name", "short_names_opt", multiple=True,
              help="Filter by short_name (repeatable; mutually exclusive with INPUT).")
@click.option("--framework-key", "framework_keys", multiple=True,
              help="Filter by framework_key (repeatable; mutually exclusive with INPUT).")
@click.option("--poly-uuid", "poly_uuids", multiple=True,
              help="Filter by poly_uuid (repeatable; mutually exclusive with INPUT).")
@click.option("--cohort", "cohorts", multiple=True,
              help="Filter by cohort (repeatable; mutually exclusive with INPUT).")
@click.option("--where", "where_sql", default=None,
              help="Raw SQL WHERE expression against tm.geoparquet (alias `p`). "
                   "Mutually exclusive with INPUT. Read-only; `;` and DDL/DML are rejected.")
@click.pass_context
def resolve(ctx, input, output, year, year_from_plantstart, geoparquet, save_polygons,
            project_ids, short_names_opt, framework_keys, poly_uuids, cohorts, where_sql):
    """Resolve any supported input to a canonical tiles CSV.

    INPUT can be:

    \b
      - Tiles CSV (Year, X, Y, X_tile, Y_tile)   -> passthrough
      - Request CSV (project_id, plantstart_year) -> query geoparquet
      - Polygon file (GeoJSON, GPKG, etc.)        -> spatial join, needs --year
      - JSON request (legacy inbound format)      -> DuckDB tile lookup
      - Short name (e.g. GHA_22_INEC)             -> query geoparquet

    INPUT may be omitted when any filter flag is provided:
    --where / --poly-uuid / --cohort / --project-id / --short-name / --framework-key.

    \b
    Examples:
        gri-ttc resolve GHA_22_INEC -o tiles.csv
        gri-ttc resolve request.csv --geoparquet temp/tm.geoparquet -o tiles.csv
        gri-ttc resolve polygons.geojson --year 2023 -o tiles.csv
        gri-ttc resolve --poly-uuid abc-123 --year 2023 -o tiles.csv
        gri-ttc resolve --where "short_name='GHA_22_INEC'" -o tiles.csv
    """
    from gri_tile_pipeline.cli_context import emit_json, get as get_ctx
    from gri_tile_pipeline.steps.resolve_input import resolve_to_tiles
    from gri_tile_pipeline.tiles.csv_io import write_tiles_csv

    gri = get_ctx(ctx)
    cfg = gri.cfg

    has_filter = bool(
        where_sql or poly_uuids or cohorts
        or project_ids or short_names_opt or framework_keys
    )
    if not input and not has_filter:
        raise click.UsageError(
            "Provide INPUT or at least one filter "
            "(--where / --poly-uuid / --cohort / --project-id / --short-name / --framework-key)."
        )
    if input and has_filter:
        raise click.UsageError("Provide INPUT or filter flags, not both.")

    resolved = resolve_to_tiles(
        input, cfg,
        year=year,
        year_from_plantstart=year_from_plantstart,
        geoparquet=geoparquet,
        where_sql=where_sql,
        poly_uuids=list(poly_uuids) or None,
        cohorts=list(cohorts) or None,
        project_ids=list(project_ids) or None,
        short_names=list(short_names_opt) or None,
        framework_keys=list(framework_keys) or None,
    )

    source_desc = input or "filter"
    if not resolved.tiles:
        logger.warning(f"No tiles resolved from {source_desc}")
        emit_json(ctx, {
            "command": "resolve", "status": "no_work", "input": input,
            "input_type": resolved.input_type, "n_tiles": 0,
        })
        ctx.exit(ExitCode.NO_WORK)
        return

    write_tiles_csv(output, resolved.tiles)
    logger.info(f"Wrote {len(resolved.tiles)} tiles to {output} ({resolved.input_type})")

    if save_polygons and resolved.polygons_gdf is not None:
        resolved.polygons_gdf.to_file(save_polygons, driver="GeoJSON")
        logger.info(f"Polygons saved to {save_polygons}")

    emit_json(ctx, {
        "command": "resolve",
        "status": "ok",
        "input": input,
        "input_type": resolved.input_type,
        "n_tiles": len(resolved.tiles),
        "output": output,
        "polygons_saved_to": save_polygons,
        "metadata": resolved.metadata,
    })


# ---------------------------------------------------------------------------
# tiles (group): missing / split / validate
# ---------------------------------------------------------------------------

@gri_ttc.group()
def tiles():
    """Tile-CSV utilities: missing, split, validate."""


@tiles.command("missing")
@click.option("--geoparquet", default="temp/tm.geoparquet", show_default=True,
              help="Path to TerraMatch geoparquet.")
@click.option("--tiledb", default=None,
              help="Path to tiledb.parquet (defaults to config pipeline.parquet_path).")
@click.option("--short-name", default=None, help="Filter by project short_name.")
@click.option("--framework-key", default=None, help="Filter by cohort framework_key.")
@click.option("-o", "--output", default=None, type=click.Path(),
              help="Output tiles CSV path. Without this, prints a summary only.")
@click.pass_context
def tiles_missing(ctx, geoparquet, tiledb, short_name, framework_key, output):
    """Find tiles for polygons missing TTC values.

    \b
    Examples:
        gri-ttc tiles missing                              # summary of what's missing
        gri-ttc tiles missing --short-name RWA_23_AEE -o missing.csv
        gri-ttc tiles missing --framework-key hbf -o hbf_missing.csv
    """
    from gri_tile_pipeline.cli_context import emit_json, get as get_ctx
    from gri_tile_pipeline.tiles.csv_io import write_tiles_csv
    from gri_tile_pipeline.tiles.missing import generate_missing_tiles, summarize_missing

    gri = get_ctx(ctx)
    tiledb = tiledb or gri.cfg.parquet_path

    if output is None and short_name is None and framework_key is None:
        summary = summarize_missing(geoparquet)
        if not gri.json_mode:
            click.echo(f"Total polygons missing TTC: {summary['total_missing']:,}\n")
            click.echo("By cohort:")
            for row in summary["by_cohort"]:
                click.echo(f"  {row['framework_key'] or '(null)':<30} "
                           f"{row['polygons_missing_ttc']:>9,}")
        emit_json(ctx, {"command": "tiles.missing", "status": "summary", **summary})
        return

    tiles_list = generate_missing_tiles(
        geoparquet, tiledb, short_name=short_name, framework_key=framework_key,
    )

    if not tiles_list:
        logger.warning("No missing-TTC tiles matched the filter.")
        emit_json(ctx, {"command": "tiles.missing", "status": "no_work", "n_tiles": 0})
        ctx.exit(ExitCode.NO_WORK)
        return

    if output:
        write_tiles_csv(output, tiles_list)
        logger.info(f"Wrote {len(tiles_list)} tiles to {output}")

    year_counts: dict[int, int] = {}
    for t in tiles_list:
        year_counts[t["year"]] = year_counts.get(t["year"], 0) + 1
    emit_json(ctx, {
        "command": "tiles.missing", "status": "ok",
        "n_tiles": len(tiles_list), "output": output,
        "by_year": year_counts,
    })


@tiles.command("split")
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--chunk-size", type=int, default=None,
              help="Fixed rows per chunk. Default: 100, 200, 400, 800, then 1600 each.")
@click.option("--encoding", default="utf-8", show_default=True,
              help="Input file encoding (try utf-8-sig for Excel exports).")
@click.pass_context
def tiles_split(ctx, csv_path, chunk_size, encoding):
    """Split CSV_PATH into header-preserving chunk files.

    Writes <base>_chunk_<n>.csv next to the input.
    """
    from gri_tile_pipeline.cli_context import emit_json
    from gri_tile_pipeline.tiles.split import split_csv

    files = split_csv(csv_path, chunk_size=chunk_size, encoding=encoding)
    for f in files:
        click.echo(f"Wrote {f}")
    emit_json(ctx, {"command": "tiles.split", "status": "ok",
                    "n_chunks": len(files), "chunks": files})


@tiles.command("validate")
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--check-s3", is_flag=True,
              help="Also ping S3 for each tile to confirm availability.")
@click.option("--dest", default=None,
              help="S3 destination (required with --check-s3).")
@click.option("--region", default=None,
              help="AWS region (defaults to config zonal.tile_region).")
@click.option("--check-type", type=click.Choice(["raw_ard", "predictions"]),
              default="predictions", show_default=True)
@click.pass_context
def tiles_validate(ctx, csv_path, check_s3, dest, region, check_type):
    """Validate CSV_PATH: required columns present, row count, optional S3 ping."""
    from gri_tile_pipeline.cli_context import emit_json, get as get_ctx
    from gri_tile_pipeline.tiles.validate import validate_tiles_csv

    gri = get_ctx(ctx)
    region = region or gri.cfg.zonal.tile_region

    report = validate_tiles_csv(
        csv_path, check_s3=check_s3, dest=dest, region=region,
        profile=gri.aws_profile, check_type=check_type,
    )

    if not gri.json_mode:
        click.echo(f"Path: {report.path}")
        click.echo(f"Rows: {report.n_rows}")
        click.echo(f"Schema OK: {report.schema_ok}")
        if report.missing_columns:
            click.echo(f"Missing columns: {report.missing_columns}")
        if report.extra_columns:
            click.echo(f"Extra columns: {report.extra_columns}")
        if report.parse_errors:
            click.echo(f"Parse errors: {report.parse_errors}")
        if report.availability:
            click.echo(f"S3: {report.availability['n_existing']} present, "
                       f"{report.availability['n_missing']} missing")

    emit_json(ctx, {"command": "tiles.validate", **report.as_dict()})
    if not report.schema_ok or report.parse_errors:
        ctx.exit(ExitCode.BAD_INPUT)


# ---------------------------------------------------------------------------
# report / audit-drops / preview-polygon
# ---------------------------------------------------------------------------

@gri_ttc.command()
@click.option("--input", "input_csv", default=None, type=click.Path(exists=True),
              help="Request CSV with project_id[,plantstart_year] rows.")
@click.option("--project-id", "project_ids", multiple=True,
              help="Filter by project_id (repeatable).")
@click.option("--short-name", "short_names", multiple=True,
              help="Filter by project short_name (repeatable).")
@click.option("--framework-key", "framework_keys", multiple=True,
              help="Filter by cohort framework_key (repeatable).")
@click.option("--poly-uuid", "poly_uuids", multiple=True,
              help="Filter by poly_uuid (repeatable).")
@click.option("--cohort", "cohorts", multiple=True,
              help="Filter by cohort membership (repeatable; uses list_contains).")
@click.option("--where", "where_sql", default=None,
              help="Raw SQL WHERE expression against tm.geoparquet (alias `p`). "
                   "Read-only: DDL/DML/`;` are rejected. Example: "
                   "\"YEAR(plantstart) = 2023 AND project_id IN ('p1','p2')\".")
@click.option("--geoparquet", default="temp/tm.geoparquet", show_default=True)
@click.option("--tiledb", default=None,
              help="Path to tiledb.parquet (defaults to config pipeline.parquet_path).")
@click.option("--bucket", default=None, help="S3 bucket (defaults to zonal.tile_bucket).")
@click.option("--region", default=None, help="AWS region (defaults to zonal.tile_region).")
@click.option("--check-type", type=click.Choice(["predictions", "raw_ard"]),
              default="predictions", show_default=True)
@click.option("--skip-s3", is_flag=True, help="Skip phases 3-4 (no AWS needed).")
@click.option("--output-dir", default=".", show_default=True,
              help="Directory for the Markdown report + missing-tiles CSV.")
@click.pass_context
def report(ctx, input_csv, project_ids, short_names, framework_keys,
           poly_uuids, cohorts, where_sql, geoparquet,
           tiledb, bucket, region, check_type, skip_s3, output_dir):
    """Generate a 4-phase TTC status report.

    \b
    Phases:
      1. Request Scope       2. TTC Coverage
      3. Tile Availability   4. Tiles to Generate

    \b
    Examples:
        gri-ttc report --input request.csv --skip-s3
        gri-ttc report --short-name RWA_23_AEE
        gri-ttc report --framework-key terrafund-landscapes --output-dir temp/
        gri-ttc report --poly-uuid abc-123 --poly-uuid def-456
        gri-ttc report --where "YEAR(plantstart) = 2023 AND country = 'GHA'"
    """
    from gri_tile_pipeline.cli_context import emit_json, get as get_ctx
    from gri_tile_pipeline.reporting.status_report import generate_report

    gri = get_ctx(ctx)
    tiledb = tiledb or gri.cfg.parquet_path
    bucket = bucket or gri.cfg.zonal.tile_bucket
    region = region or gri.cfg.zonal.tile_region

    if not (input_csv or project_ids or short_names or framework_keys
            or poly_uuids or cohorts or where_sql):
        raise click.UsageError(
            "Provide at least one of --input, --project-id, --short-name, "
            "--framework-key, --poly-uuid, --cohort, --where.",
        )

    result = generate_report(
        input_csv=input_csv,
        project_ids=list(project_ids),
        short_names=list(short_names),
        framework_keys=list(framework_keys),
        poly_uuids=list(poly_uuids),
        cohorts=list(cohorts),
        where_sql=where_sql,
        geoparquet=geoparquet,
        tiledb=tiledb,
        bucket=bucket,
        region=region,
        check_type=check_type,
        skip_s3=skip_s3,
        aws_profile=gri.aws_profile,
        output_dir=output_dir,
        progress=None if gri.json_mode else (lambda m: click.echo(m)),
    )

    if not gri.json_mode:
        click.echo(f"\nReport: {result.report_path}")
        if result.tiles_csv_path:
            click.echo(f"Tiles CSV: {result.tiles_csv_path}")

    emit_json(ctx, {
        "command": "report", "status": "ok",
        "report_path": result.report_path,
        "tiles_csv_path": result.tiles_csv_path,
        "scope": {"total_projects": result.scope["total_projects"],
                  "total_polygons": result.scope["total_polygons"]},
        "coverage": {k: result.coverage[k]
                     for k in ("total", "correct_yr", "wrong_yr", "missing")},
        "tile_avail": (None if result.tile_avail is None else
                       {k: result.tile_avail.get(k)
                        for k in ("total_needed", "existing", "missing")}),
    })


@gri_ttc.command("audit-drops")
@click.option("--request", "request_csv", required=True, type=click.Path(exists=True),
              help="Input request CSV (project_id, plantstart_year).")
@click.option("--stats", "stats_csv", required=True, type=click.Path(exists=True),
              help="Output stats CSV from the pipeline run.")
@click.option("--geoparquet", default="temp/tm.geoparquet", show_default=True)
@click.option("-o", "--output", default=None, type=click.Path(),
              help="Output CSV for the dropped-polygon report.")
@click.pass_context
def audit_drops(ctx, request_csv, stats_csv, geoparquet, output):
    """Audit polygons that were dropped during a pipeline run, classify by cause."""
    import pandas as pd
    from gri_tile_pipeline.cli_context import emit_json, get as get_ctx
    from gri_tile_pipeline.reporting.audit import audit_drops as run_audit

    gri = get_ctx(ctx)
    report_result = run_audit(request_csv, stats_csv, geoparquet)

    if not gri.json_mode:
        click.echo(f"Expected: {report_result.n_expected}")
        click.echo(f"Produced: {report_result.n_produced}")
        click.echo(f"Dropped:  {report_result.n_dropped}")
        if report_result.n_dropped:
            click.echo("\nStatus breakdown:")
            for status, count in report_result.status_counts.items():
                click.echo(f"  {status}: {count}")

    if report_result.rows:
        out_path = output or "dropped_polygons_report.csv"
        pd.DataFrame(report_result.rows).to_csv(out_path, index=False)
        if not gri.json_mode:
            click.echo(f"\nReport written to {out_path}")
    else:
        out_path = None

    emit_json(ctx, {
        "command": "audit-drops", "status": "ok",
        "n_expected": report_result.n_expected,
        "n_produced": report_result.n_produced,
        "n_dropped": report_result.n_dropped,
        "status_counts": report_result.status_counts,
        "output": out_path,
    })


@gri_ttc.command("preview-polygon")
@click.argument("poly_uuid")
@click.option("--year", type=int, default=None,
              help="Prediction year (default: plantstart - 1).")
@click.option("--geoparquet", default="temp/tm.geoparquet", show_default=True)
@click.option("--show-shifts", is_flag=True,
              help="Overlay 8-directional ~10m shift boundaries.")
@click.option("-o", "--output", default=None, type=click.Path(),
              help="Output PNG path (default: temp/<poly_uuid>_ttc_preview.png).")
@click.pass_context
def preview_polygon(ctx, poly_uuid, year, geoparquet, show_shifts, output):
    """Render a PNG preview of POLY_UUID overlaid on its TTC prediction tiles."""
    from gri_tile_pipeline.cli_context import emit_json, get as get_ctx
    from gri_tile_pipeline.reporting.preview import preview_polygon as run_preview

    gri = get_ctx(ctx)
    out_path = run_preview(
        poly_uuid, gri.cfg,
        year=year, geoparquet=geoparquet, output=output, show_shifts=show_shifts,
    )
    if not gri.json_mode:
        click.echo(f"Preview saved -> {out_path}")
    emit_json(ctx, {
        "command": "preview-polygon", "status": "ok",
        "poly_uuid": poly_uuid, "output": out_path,
    })


# ---------------------------------------------------------------------------
# runs (group): list / show / failed / retry
# ---------------------------------------------------------------------------

@gri_ttc.group()
def runs():
    """Query saved run history (list, show, failed, retry)."""


@runs.command("list")
@click.option("--limit", type=int, default=20, show_default=True,
              help="Max number of runs to show.")
@click.pass_context
def runs_list(ctx, limit):
    """Tabular list of past runs."""
    from gri_tile_pipeline.cli_context import emit_json, get as get_ctx
    from gri_tile_pipeline.tracking.run_index import list_runs

    gri = get_ctx(ctx)
    summaries = list_runs(gri.run_history_dir)[:limit]

    if not gri.json_mode:
        if not summaries:
            click.echo(f"No runs under {gri.run_history_dir}/")
        else:
            click.echo(f"{'run_id':<16} {'when':<20} {'step':<10} "
                       f"{'jobs':>6} {'ok':>6} {'fail':>6}")
            for s in summaries:
                click.echo(
                    f"{s.run_id:<16} {(s.start_time or '')[:19]:<20} "
                    f"{(s.step or '-'):<10} {s.n_jobs:>6} "
                    f"{s.n_success:>6} {s.n_failed:>6}"
                )

    emit_json(ctx, {
        "command": "runs.list", "status": "ok",
        "run_history_dir": gri.run_history_dir,
        "runs": [s.as_dict() for s in summaries],
    })


@runs.command("show")
@click.argument("run_id")
@click.pass_context
def runs_show(ctx, run_id):
    """Show a run's full summary."""
    from gri_tile_pipeline.cli_context import emit_json, get as get_ctx
    from gri_tile_pipeline.tracking.run_index import get_run

    gri = get_ctx(ctx)
    try:
        summary = get_run(gri.run_history_dir, run_id)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))

    if not gri.json_mode:
        click.echo(f"Run: {summary.run_id}")
        click.echo(f"  Step:     {summary.step}")
        click.echo(f"  Started:  {summary.start_time}")
        click.echo(f"  Ended:    {summary.end_time}")
        click.echo(f"  Duration: {summary.duration_sec:.1f}s"
                   if summary.duration_sec else "  Duration: -")
        click.echo(f"  Jobs:     {summary.n_jobs} "
                   f"(ok={summary.n_success}, partial={summary.n_partial}, "
                   f"fail={summary.n_failed})")
        click.echo(f"  Failed tiles: {summary.n_failed_tiles}")
        if summary.by_task_type:
            click.echo("  By task type:")
            for task, counts in summary.by_task_type.items():
                click.echo(f"    {task}: " + ", ".join(
                    f"{k}={v}" for k, v in counts.items() if v
                ))
        click.echo(f"\n  Report: {summary.report_md}")
        click.echo(f"  Jobs CSV: {summary.jobs_csv}")
        click.echo(f"  Failed CSV: {summary.failed_csv}")

    emit_json(ctx, {"command": "runs.show", "status": "ok",
                    **summary.as_dict()})


@runs.command("failed")
@click.argument("run_id")
@click.option("-o", "--output", default=None, type=click.Path(),
              help="Copy failed.csv to this path (default: stdout of path).")
@click.pass_context
def runs_failed(ctx, run_id, output):
    """Emit the tiles CSV of failed tiles from a past run (pipe into `gri-ttc run`)."""
    import shutil

    from gri_tile_pipeline.cli_context import emit_json, get as get_ctx
    from gri_tile_pipeline.tracking.run_index import get_run

    gri = get_ctx(ctx)
    try:
        summary = get_run(gri.run_history_dir, run_id)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))

    if not summary.failed_csv.exists():
        raise click.ClickException(f"No failed.csv for run {run_id!r}")

    out_path = str(summary.failed_csv)
    if output:
        shutil.copyfile(summary.failed_csv, output)
        out_path = output

    if not gri.json_mode:
        click.echo(out_path)

    emit_json(ctx, {"command": "runs.failed", "status": "ok",
                    "run_id": run_id, "output": out_path,
                    "n_failed_tiles": summary.n_failed_tiles})


@runs.command("retry")
@click.argument("run_id")
@click.option("-o", "--output", default=None, type=click.Path(),
              help="Write failed tiles to this path (default: <run_dir>/failed.csv).")
@click.pass_context
def runs_retry(ctx, run_id, output):
    """Print the command to re-run failed tiles from RUN_ID.

    Does not execute — the user wires the failed CSV into `gri-ttc run` with
    their chosen --dest, --steps, etc. This keeps the retry explicit and safe.
    """
    from gri_tile_pipeline.cli_context import emit_json, get as get_ctx
    from gri_tile_pipeline.tracking.run_index import get_run

    gri = get_ctx(ctx)
    try:
        summary = get_run(gri.run_history_dir, run_id)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))

    if not summary.failed_csv.exists() or summary.n_failed_tiles == 0:
        msg = f"Run {run_id!r} has no failed tiles to retry."
        if not gri.json_mode:
            click.echo(msg)
        emit_json(ctx, {"command": "runs.retry", "status": "no_work",
                        "run_id": run_id})
        return

    out_path = str(summary.failed_csv)
    if output:
        import shutil
        shutil.copyfile(summary.failed_csv, output)
        out_path = output

    step = summary.step or "download,predict"
    suggested = (
        f"gri-ttc run {out_path} --dest <your-dest> "
        f"--steps {step} --yes"
    )

    if not gri.json_mode:
        click.echo(f"Failed tiles: {out_path} ({summary.n_failed_tiles} tiles)")
        click.echo(f"Suggested:    {suggested}")

    emit_json(ctx, {"command": "runs.retry", "status": "ok",
                    "run_id": run_id, "failed_csv": out_path,
                    "n_failed_tiles": summary.n_failed_tiles,
                    "suggested_command": suggested})


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

_TILE_ID_RE = __import__("re").compile(r"^(\d+)X(\d+)Y$")


@gri_ttc.command()
@click.argument("input_path")
@click.option("--dest", default=None, help="S3 destination (s3://bucket/prefix). "
              "Defaults to s3://<zonal.tile_bucket> for tile-ID input.")
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
      - Tile ID (e.g. 1000X871Y)                   -> prints per-source report
      - Tiles CSV (Year, X, Y, X_tile, Y_tile)     -> checks directly
      - Request CSV (project_id, plantstart_year)  -> resolves via geoparquet
      - Polygon file (GeoJSON, GeoPackage, etc.)   -> spatial join, needs --year
      - JSON request (legacy inbound format)       -> resolves via tile lookup

    \b
    Examples:
        gri-ttc check 1000X871Y --year 2023
        gri-ttc check tiles.csv --dest s3://tof-output --check-type raw_ard
        gri-ttc check request.csv --dest s3://tof-output --geoparquet temp/tm.geoparquet
        gri-ttc check polygons.geojson --dest s3://tof-output --year 2023
    """
    import os as _os
    from datetime import datetime as _dt

    from gri_tile_pipeline.cli_context import emit_json
    from gri_tile_pipeline.steps.resolve_input import resolve_to_tiles
    from gri_tile_pipeline.tiles.availability import (
        AVAILABLE_SOURCES,
        check_availability,
        check_availability_by_source,
        format_tile_report,
    )
    from gri_tile_pipeline.tiles.csv_io import write_tiles_csv

    cfg = ctx.obj["cfg"]
    region = region or cfg.zonal.tile_region

    tile_match = _TILE_ID_RE.match(input_path)
    if tile_match:
        x_tile, y_tile = int(tile_match.group(1)), int(tile_match.group(2))
        resolved_year = year if year is not None else _dt.utcnow().year
        resolved_dest = dest or f"s3://{cfg.zonal.tile_bucket}"
        tile = {"year": resolved_year, "X_tile": x_tile, "Y_tile": y_tile}

        presence_map = check_availability_by_source(
            [tile], resolved_dest, sources=AVAILABLE_SOURCES, region=region,
        )
        presence = presence_map[(resolved_year, x_tile, y_tile)]

        click.echo(format_tile_report(tile, presence, resolved_dest))
        emit_json(ctx, {
            "command": "check",
            "tile": f"{x_tile}X{y_tile}Y",
            "year": resolved_year,
            "dest": resolved_dest,
            "sources": presence,
        })
        n_missing = sum(1 for v in presence.values() if not v)
        if n_missing and exit_on_missing:
            ctx.exit(ExitCode.TILES_MISSING)
        ctx.exit(ExitCode.SUCCESS)
        return

    if not _os.path.exists(input_path):
        raise click.UsageError(
            f"INPUT_PATH '{input_path}' is not a tile ID (e.g. 1000X871Y) "
            "and does not exist as a file."
        )
    if dest is None:
        raise click.UsageError("--dest is required for file-path input.")

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
    run_id = ctx.obj.get("run_id")

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
            run_id=run_id,
        )
    else:
        from gri_tile_pipeline.steps.predict import run_predict
        tracker = run_predict(
            tiles_csv, dest, cfg,
            model_path=model_path,
            memory_mb=mem, retries=retries,
            skip_existing=skip_existing,
            report_dir=report_dir, debug=debug,
            run_id=run_id,
        )

    ctx.exit(exit_code_from_tracker(tracker))


# ---------------------------------------------------------------------------
# stats (stub — Phase 4)
# ---------------------------------------------------------------------------

@gri_ttc.command()
@click.argument("polygons", type=click.Path(exists=True))
@click.option("--dest", "--tiles-bucket", "dest", default=None,
              help="S3 bucket (or local path) containing prediction tiles. "
                   "--tiles-bucket kept as alias for backwards compatibility.")
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
def stats(ctx, polygons, dest, year, output, lookup_parquet, lookup_csv, include_cols, lulc_raster, shift_error):
    """Calculate zonal tree cover statistics for polygons."""
    from gri_tile_pipeline.steps.zonal_stats import run_zonal_stats

    cfg = ctx.obj["cfg"]
    bucket = dest or cfg.zonal.tile_bucket
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

    from gri_tile_pipeline.cli_context import emit_json
    emit_json(ctx, {
        "command": "stats", "status": "ok",
        "polygons": polygons, "year": year, "output": output,
        "dest": bucket,
    })
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
                    run_id=ctx.obj.get("run_id"),
                )
            else:
                from gri_tile_pipeline.steps.predict import run_predict
                pred_tracker = run_predict(
                    predict_csv, dest, cfg, skip_existing=skip_existing,
                    run_id=ctx.obj.get("run_id"),
                )
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
# tm-patch
# ---------------------------------------------------------------------------

@gri_ttc.command("tm-patch")
@click.option("--results", "results_csv", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to results CSV (output of `gri-ttc stats`).")
@click.option("--project-id", required=True,
              help="TerraMatch projectId to scope the patch to.")
@click.option("--env", "tm_env", type=click.Choice(["staging", "production"]),
              default="staging", show_default=True,
              help="TerraMatch environment.")
@click.option("--indicator-slug", default="treeCover", show_default=True,
              help="indicatorSlug sent to TerraMatch.")
@click.option("--year", type=int, default=None,
              help="yearOfAnalysis fallback; if omitted, rows must populate --year-column.")
@click.option("--year-column", default="year", show_default=True,
              help="Column in results.csv that holds the per-row year (falls back to --year).")
@click.option("--project-phase", default="implementation", show_default=True,
              help="projectPhase sent to TerraMatch.")
@click.option("--percent-column", default="ttc", show_default=True,
              help="Column in results.csv that holds the percent-cover value.")
@click.option("--uncertainty-column", default=None,
              help="Column for plusMinusPercent. Default: auto-detect from CSV.")
@click.option("--apply/--dry-run", default=False,
              help="Default is dry-run; pass --apply to actually PATCH.")
@click.option("--limit", type=int, default=None,
              help="Stop after N rows (smoke testing).")
@click.option("--base-url", default=None, help="Override TerraMatch base URL.")
@click.option("--token", default=None,
              help="Override TerraMatch bearer token (prefer GRI_TM_TOKEN env var).")
@click.option("-o", "--report", "report_csv", default=None, type=click.Path(),
              help="Write per-row outcome CSV here.")
@click.pass_context
def tm_patch(ctx, results_csv, project_id, tm_env, indicator_slug, year,
             year_column, project_phase, percent_column, uncertainty_column,
             apply, limit, base_url, token, report_csv):
    """Patch TTC stats from a results CSV back onto TerraMatch polygons.

    Matches each row's ``poly_uuid`` against polygon ids returned by
    ``GET /sitePolygons`` for ``--project-id``. Unmatched rows are reported
    but not patched. Default is dry-run — pass ``--apply`` to actually send.

    \b
    Examples:
        gri-ttc tm-patch --results results.csv --project-id <TM_PROJ> --year 2023
        gri-ttc tm-patch --results results.csv --project-id <TM_PROJ> --year 2023 \\
            --env staging --apply --limit 5
        gri-ttc tm-patch --results results.csv --project-id <TM_PROJ> --year 2023 \\
            --env production --apply -o patched.csv
    """
    import pandas as pd

    from gri_tile_pipeline.cli_context import emit_json, get as get_ctx
    from gri_tile_pipeline.terramatch import (
        IndicatorSpec,
        TMClient,
        detect_uncertainty_column,
        load_results,
        resolve_tm_creds,
        run_patch,
        summarize,
    )
    from gri_tile_pipeline.terramatch.secrets import MissingTMCredential

    gri = get_ctx(ctx)

    try:
        base_url_resolved, token_resolved = resolve_tm_creds(
            tm_env, token=token, base_url=base_url,
        )
    except MissingTMCredential as e:
        raise click.ClickException(str(e))

    rows, columns = load_results(results_csv)
    if not rows:
        logger.warning(f"{results_csv} has no rows")
        emit_json(ctx, {"command": "tm-patch", "status": "no_work",
                        "results": results_csv})
        ctx.exit(ExitCode.NO_WORK)
        return

    unc_col = uncertainty_column or detect_uncertainty_column(columns)
    if unc_col:
        logger.info(f"Uncertainty column: {unc_col}")
    else:
        logger.info("No uncertainty column detected; plusMinusPercent will be omitted")

    spec = IndicatorSpec(
        year=year,
        slug=indicator_slug,
        project_phase=project_phase,
        year_column=year_column if year_column in columns else None,
        percent_column=percent_column,
        uncertainty_column=unc_col,
    )

    mode = "APPLY" if apply else "dry-run"
    click.echo(
        f"tm-patch [{mode}] project={project_id} env={tm_env} "
        f"rows={len(rows)} slug={spec.slug} year={spec.year}"
    )
    if apply and tm_env == "production" and not gri.yes:
        click.confirm(
            f"Send {min(len(rows), limit or len(rows))} PATCHes to PRODUCTION?",
            abort=True,
        )

    client = TMClient(base_url_resolved, token_resolved)

    try:
        outcomes = run_patch(
            rows, project_id, client, spec, apply=apply, limit=limit,
        )
    except RuntimeError as e:
        logger.error(str(e))
        raise click.ClickException(str(e))

    counts = summarize(outcomes)
    if not gri.json_mode:
        click.echo(
            f"\nSummary: total={counts['total']} sent={counts.get('sent', 0)} "
            f"dryrun={counts.get('dryrun', 0)} "
            f"unmatched={counts.get('unmatched', 0)} "
            f"error={counts.get('error', 0)}"
        )

    if report_csv:
        pd.DataFrame([o.as_dict() for o in outcomes]).to_csv(report_csv, index=False)
        logger.info(f"Outcome report: {report_csv}")

    emit_json(ctx, {
        "command": "tm-patch",
        "status": "ok",
        "apply": apply,
        "env": tm_env,
        "project_id": project_id,
        "counts": counts,
        "report": report_csv,
    })

    n_error = counts.get("error", 0)
    n_sent_or_dry = counts.get("sent", 0) + counts.get("dryrun", 0)
    if n_error and not n_sent_or_dry:
        ctx.exit(ExitCode.TOTAL_FAILURE)
    elif n_error:
        ctx.exit(ExitCode.PARTIAL_FAILURE)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

@gri_ttc.command()
@click.option("--check-tm/--skip-tm", default=False,
              help="Also verify the TerraMatch API is reachable with the current token.")
@click.option("--tm-env", type=click.Choice(["staging", "production"]),
              default="staging", show_default=True)
@click.pass_context
def doctor(ctx, check_tm, tm_env):
    """Verify the local environment is ready to run the pipeline.

    Runs a series of independent checks (config, AWS creds, Lithops env,
    geoparquet reachability, optional TerraMatch API) and reports the result
    of each. Non-zero exit if any required check fails.
    """
    from gri_tile_pipeline.cli_context import emit_json, get as get_ctx
    from gri_tile_pipeline.diagnostics import run_checks

    gri = get_ctx(ctx)
    results = run_checks(
        cfg=gri.cfg, aws_profile=gri.aws_profile,
        check_tm=check_tm, tm_env=tm_env,
    )

    if not gri.json_mode:
        width = max(len(r.name) for r in results)
        for r in results:
            mark = "OK  " if r.ok else "FAIL"
            click.echo(f"  [{mark}] {r.name:<{width}}  {r.detail}")
            if r.hint and not r.ok:
                click.echo(f"          hint: {r.hint}")

    any_failed = any(not r.ok for r in results)
    emit_json(ctx, {
        "command": "doctor",
        "status": "fail" if any_failed else "ok",
        "checks": [r.as_dict() for r in results],
    })
    if any_failed:
        ctx.exit(ExitCode.BAD_INPUT)


# ---------------------------------------------------------------------------
# run-project
# ---------------------------------------------------------------------------

@gri_ttc.command("run-project")
@click.argument("short_name", required=False, default=None)
@click.option("--input", "input_csv", default=None, type=click.Path(exists=True),
              help="Request CSV with project_id,plantstart_year columns.")
@click.option("--project-id", "project_ids", multiple=True,
              help="Filter by project_id (repeatable).")
@click.option("--short-name", "short_names_opt", multiple=True,
              help="Filter by project short_name (repeatable). Alternative to the SHORT_NAME positional.")
@click.option("--framework-key", "framework_keys", multiple=True,
              help="Filter by cohort framework_key (repeatable).")
@click.option("--poly-uuid", "poly_uuids", multiple=True,
              help="Filter by poly_uuid (repeatable).")
@click.option("--cohort", "cohorts", multiple=True,
              help="Filter by cohort membership (repeatable; uses list_contains).")
@click.option("--where", "where_sql", default=None,
              help="Raw SQL WHERE expression against tm.geoparquet (alias `p`). "
                   "Read-only; DDL/DML/`;` are rejected.")
@click.option("--dest", required=True, envvar="DEST",
              help="S3 URI (s3://bucket) or local path for ARD + predictions.")
@click.option("--geoparquet", default="temp/tm.geoparquet",
              help="Path to TerraMatch geoparquet file.")
@click.option("--year", type=int, default=None,
              help="Override prediction year (default: plantstart - 1).")
@click.option("-o", "--output", default=None,
              help="Output CSV path.")
@click.option("--local", is_flag=True,
              help="Run download + predict workers in-process on this machine "
                   "instead of fanning out via Lithops/Lambda. Default: Lithops/Lambda.")
@click.option("--max-workers", type=int, default=1,
              help="Parallel workers for --local mode (ignored when running via Lithops).")
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
@click.option("--tm-patch", "tm_patch_enabled", is_flag=True,
              help="After stats, patch results back to TerraMatch.")
@click.option("--tm-patch-env", type=click.Choice(["staging", "production"]),
              default="staging", show_default=True,
              help="Environment for --tm-patch.")
@click.option("--tm-patch-project-id", default=None,
              help="TerraMatch projectId to patch. Required with --tm-patch.")
@click.option("--tm-patch-apply", is_flag=True,
              help="Actually PATCH (default: dry-run).")
@click.pass_context
def run_project(ctx, short_name, input_csv, project_ids, short_names_opt,
                framework_keys, poly_uuids, cohorts, where_sql,
                dest, geoparquet, year, output,
                local, max_workers, skip_existing, lulc_raster, shift_error,
                missing_only, check_only, dry_run, yes,
                tm_patch_enabled, tm_patch_env, tm_patch_project_id,
                tm_patch_apply):
    """End-to-end pipeline for a TerraMatch project.

    Provide one of:
      - SHORT_NAME positional argument, or
      - --input with a request CSV (project_id, plantstart_year), or
      - one or more filter flags: --where / --poly-uuid / --cohort /
        --project-id / --short-name / --framework-key.

    \b
    Execution mode:
        Default: download + predict fan out via Lithops/AWS Lambda.
                 LITHOPS_ENV must be set (e.g. `LITHOPS_ENV=datalab-test`).
        --local: run both steps in-process on this machine (slow; useful
                 for smoke-testing without cloud credentials).

    \b
    Examples:
        LITHOPS_ENV=datalab-test gri-ttc run-project GHA_22_INEC --dest s3://tof-output --yes
        gri-ttc run-project --input request.csv --dest s3://tof-output --dry-run
        gri-ttc run-project --short-name GHA_22_INEC --short-name RWA_23_AEE --dest s3://tof-output
        gri-ttc run-project --poly-uuid abc-123 --year 2023 --dest s3://tof-output --local
        gri-ttc run-project --where "country='GHA' AND YEAR(plantstart)=2023" --dest s3://tof-output
    """
    from gri_tile_pipeline.steps.project_e2e import run_project_pipeline

    has_filter = bool(
        where_sql or poly_uuids or cohorts
        or project_ids or short_names_opt or framework_keys
    )
    sources = sum([bool(short_name), bool(input_csv), has_filter])
    if sources == 0:
        raise click.UsageError(
            "Provide SHORT_NAME, --input, or at least one filter flag "
            "(--where / --poly-uuid / --cohort / --project-id / --short-name / --framework-key).",
        )
    if sources > 1:
        raise click.UsageError(
            "Provide exactly one of: SHORT_NAME, --input, or filter flags.",
        )

    cfg = ctx.obj["cfg"]

    if shift_error is not None:
        cfg.zonal.shift_error_enabled = shift_error

    if not dry_run and not check_only and not yes:
        if has_filter:
            source = "filter"
        else:
            source = short_name or input_csv
        mode_str = f"LOCAL (max_workers={max_workers})" if local else "LITHOPS/AWS Lambda"
        click.confirm(
            f"Run full pipeline for '{source}' in mode={mode_str}? "
            f"(dest={dest}, geoparquet={geoparquet})",
            abort=True,
        )

    if tm_patch_enabled and not tm_patch_project_id:
        raise click.UsageError("--tm-patch requires --tm-patch-project-id.")

    try:
        result = run_project_pipeline(
            short_name,
            dest,
            cfg,
            input_csv=input_csv,
            geoparquet=geoparquet,
            year=year,
            output=output,
            local=local,
            max_workers=max_workers,
            skip_existing=skip_existing,
            lulc_raster=lulc_raster,
            missing_only=missing_only,
            check_only=check_only,
            dry_run=dry_run,
            where_sql=where_sql,
            poly_uuids=list(poly_uuids) or None,
            cohorts=list(cohorts) or None,
            project_ids=list(project_ids) or None,
            short_names=list(short_names_opt) or None,
            framework_keys=list(framework_keys) or None,
        )
    except ValueError as e:
        logger.error(str(e))
        ctx.exit(ExitCode.BAD_INPUT)
        return

    run_exit = result["exit_code"]
    results_path = result.get("output_path")

    if (
        tm_patch_enabled
        and run_exit == ExitCode.SUCCESS
        and results_path
        and not dry_run
        and not check_only
    ):
        _invoke_tm_patch(
            ctx,
            results_csv=results_path,
            project_id=tm_patch_project_id,
            tm_env=tm_patch_env,
            year=year,
            apply=tm_patch_apply,
        )

    ctx.exit(run_exit)


def _invoke_tm_patch(ctx, *, results_csv, project_id, tm_env, year, apply):
    """Helper: run tm-patch inline after a successful run-project stats step."""
    from gri_tile_pipeline.terramatch import (
        IndicatorSpec,
        TMClient,
        detect_uncertainty_column,
        load_results,
        resolve_tm_creds,
        run_patch,
        summarize,
    )
    from gri_tile_pipeline.terramatch.secrets import MissingTMCredential

    try:
        base_url, token = resolve_tm_creds(tm_env)  # type: ignore[arg-type]
    except MissingTMCredential as e:
        logger.error(f"tm-patch skipped: {e}")
        return

    rows, columns = load_results(results_csv)
    if not rows:
        logger.warning(f"tm-patch skipped: {results_csv} has no rows")
        return

    year_column = "pred_year" if "pred_year" in columns else (
        "year" if "year" in columns else None
    )
    unc_col = detect_uncertainty_column(columns)
    spec = IndicatorSpec(
        year=year,
        year_column=year_column,
        uncertainty_column=unc_col,
    )

    logger.info(
        f"tm-patch [{'APPLY' if apply else 'dry-run'}] "
        f"project={project_id} env={tm_env} rows={len(rows)}"
    )
    client = TMClient(base_url, token)
    outcomes = run_patch(rows, project_id, client, spec, apply=apply)
    counts = summarize(outcomes)
    logger.info(
        f"tm-patch summary: total={counts['total']} "
        f"sent={counts.get('sent', 0)} dryrun={counts.get('dryrun', 0)} "
        f"unmatched={counts.get('unmatched', 0)} error={counts.get('error', 0)}"
    )
