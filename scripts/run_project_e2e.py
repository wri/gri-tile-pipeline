#!/usr/bin/env python
"""End-to-end TTC pipeline for a TerraMatch project.

Extracts polygons by short_name or request CSV, identifies required tiles,
downloads ARD, runs inference (via Lithops/Lambda by default; pass --local
for in-process), and computes zonal TTC statistics.

Usage:
    uv run python scripts/run_project_e2e.py GHA_22_INEC --dest s3://tof-output
    uv run python scripts/run_project_e2e.py --input request.csv --dest s3://tof-output
    uv run python scripts/run_project_e2e.py GHA_22_INEC --dest ./local_tiles --dry-run

Equivalent CLI commands:
    gri-ttc run-project GHA_22_INEC --dest s3://tof-output --yes
    gri-ttc run-project --input request.csv --dest s3://tof-output --yes
"""
from __future__ import annotations

import argparse
import sys

from loguru import logger

from gri_tile_pipeline.config import load_config
from gri_tile_pipeline.steps.project_e2e import run_project_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end TTC pipeline for a TerraMatch project",
    )
    parser.add_argument("short_name", nargs="?", default=None,
                        help="Project short_name (e.g. GHA_22_INEC)")
    parser.add_argument("--input", dest="input_csv", default=None,
                        help="Request CSV with project_id,plantstart_year columns")
    parser.add_argument("--dest", required=True,
                        help="S3 URI (s3://bucket) or local path for ARD + predictions")
    parser.add_argument("--geoparquet", default="temp/tm.geoparquet",
                        help="Path to TerraMatch geoparquet file")
    parser.add_argument("--year", type=int, default=None,
                        help="Override prediction year (default: plantstart - 1)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output CSV path")
    parser.add_argument("--local", action="store_true", default=False,
                        help="Run download + predict in-process on this machine. "
                             "Default: fan out via Lithops/AWS Lambda.")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Parallel workers for --local mode (ignored in Lithops mode)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip tiles already at dest (default)")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="Re-generate all tiles even if they exist")
    parser.add_argument("--lulc-raster", default=None,
                        help="Path or S3 URI to LULC raster for error propagation")
    parser.add_argument("--shift-error", action="store_true", default=False,
                        help="Enable shift error calculation (expensive)")
    parser.add_argument("--missing-only", action="store_true",
                        help="Only process polygons missing TTC for their prediction year")
    parser.add_argument("--check-only", action="store_true",
                        help="Stop after step 4: write missing tiles CSV and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without executing")
    parser.add_argument("--config", default=None,
                        help="Path to pipeline YAML config")
    args = parser.parse_args()

    if not args.short_name and not args.input_csv:
        parser.error("Provide either SHORT_NAME or --input CSV")
    if args.short_name and args.input_csv:
        parser.error("Provide SHORT_NAME or --input, not both")

    cfg = load_config(args.config)
    if args.shift_error:
        cfg.zonal.shift_error_enabled = True

    try:
        result = run_project_pipeline(
            args.short_name,
            args.dest,
            cfg,
            input_csv=args.input_csv,
            geoparquet=args.geoparquet,
            year=args.year,
            output=args.output,
            local=args.local,
            max_workers=args.max_workers,
            skip_existing=args.skip_existing,
            lulc_raster=args.lulc_raster,
            missing_only=args.missing_only,
            check_only=args.check_only,
            dry_run=args.dry_run,
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(3)

    sys.exit(result["exit_code"])


if __name__ == "__main__":
    main()
