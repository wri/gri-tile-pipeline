#!/usr/bin/env python
"""Download ARD from S3 and run local predictions for missing tiles.

Takes a tiles CSV (Year,X,Y,Y_tile,X_tile), downloads ARD from S3, runs
TF inference locally, and prints aws s3 cp commands for uploading results.

Usage:
    # Dry-run — show what would be done:
    uv run python scripts/run_missing_predictions.py temp/missing_only.csv --dry-run

    # Run predictions locally:
    uv run python scripts/run_missing_predictions.py temp/missing_only.csv \\
        --aws-profile <profile> \\
        --output-dir output/predictions

    # Keep downloaded ARD for inspection:
    uv run python scripts/run_missing_predictions.py temp/missing_only.csv \\
        --aws-profile <profile> --keep-ard
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import traceback

# Add src/ and repo root to path for imports
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, _root)

from gri_tile_pipeline.storage.tile_paths import prediction_key, raw_ard_keys
from gri_tile_pipeline.tiles.csv_io import read_tiles_csv


def _make_s3_store(bucket: str, region: str, profile: str | None = None):
    """Build an obstore S3Store with boto3 credential handling."""
    import boto3
    from obstore.auth.boto3 import Boto3CredentialProvider
    from obstore.store import S3Store

    session = boto3.Session(profile_name=profile)
    credential_provider = Boto3CredentialProvider(session)
    return S3Store(bucket, region=region, credential_provider=credential_provider)


def validate_aws(store) -> None:
    """Verify AWS credentials work by issuing a head call."""
    import obstore as obs

    try:
        obs.head(store, "__credential_check__")
    except FileNotFoundError:
        pass  # expected — credentials valid, key doesn't exist
    except Exception as e:
        first_line = str(e).split("\n")[0]
        print("AWS credential check failed:")
        print(f"  {first_line}")
        print()
        print("Make sure you have valid AWS credentials. Options:")
        print("  --aws-profile <profile>   Set an AWS profile")
        print("  aws sso login --profile <name>  If using SSO")
        sys.exit(1)


def download_ard_for_tile(
    s3_store, output_dir: str, year: int, x_tile: int, y_tile: int, *, skip_existing: bool = False,
) -> list[str]:
    """Download 6 ARD HKL files from S3 to local directory structure.

    Returns list of downloaded S3 keys.
    """
    import obstore as obs

    keys = raw_ard_keys(year, x_tile, y_tile)
    for key in keys:
        local_path = os.path.join(output_dir, key)
        if skip_existing and os.path.isfile(local_path):
            continue
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        data = obs.get(s3_store, key).bytes()
        with open(local_path, "wb") as f:
            f.write(data)
        print(f"    downloaded {key} ({len(data) / 1024:.0f} KB)")
    return keys


def check_local_ard(output_dir: str, year: int, x_tile: int, y_tile: int) -> list[str]:
    """Check which ARD files already exist locally.

    Returns list of keys that are present.
    """
    keys = raw_ard_keys(year, x_tile, y_tile)
    return [k for k in keys if os.path.isfile(os.path.join(output_dir, k))]


def cleanup_ard(output_dir: str, year: int, x_tile: int, y_tile: int) -> None:
    """Remove downloaded ARD files for a tile."""
    raw_dir = os.path.join(output_dir, str(year), "raw", str(x_tile), str(y_tile))
    if os.path.isdir(raw_dir):
        shutil.rmtree(raw_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download ARD from S3 and run local predictions for missing tiles."
    )
    parser.add_argument("csv", help="Path to tiles CSV (Year,X,Y,Y_tile,X_tile)")
    parser.add_argument(
        "--bucket", default="wri-restoration-geodata-ttc", help="S3 bucket containing ARD (default: wri-restoration-geodata-ttc)"
    )
    parser.add_argument(
        "--region", default="us-east-1", help="S3 bucket region (default: us-east-1)"
    )
    parser.add_argument("--aws-profile", default=None, help="AWS profile name")
    parser.add_argument(
        "--model-path",
        default=os.path.join(_root, "models"),
        help="Local directory containing predict_graph-172.pb",
    )
    parser.add_argument(
        "--output-dir", default="output/predictions", help="Local working directory (default: output/predictions)"
    )
    parser.add_argument(
        "--keep-ard", action="store_true", help="Keep downloaded ARD files after prediction"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without executing"
    )
    args = parser.parse_args()

    tiles = read_tiles_csv(args.csv)
    if not tiles:
        print(f"No tiles found in '{args.csv}'")
        sys.exit(1)

    # Summarise
    year_counts: dict[int, int] = {}
    for t in tiles:
        year_counts[t["year"]] = year_counts.get(t["year"], 0) + 1

    print(f"Input:      {args.csv}")
    print(f"Tiles:      {len(tiles)}")
    print(f"Years:      {', '.join(f'{y} ({c})' for y, c in sorted(year_counts.items()))}")
    print(f"S3 bucket:  s3://{args.bucket}")
    print(f"Model:      {args.model_path}")
    print(f"Output dir: {args.output_dir}")
    print()

    if args.dry_run:
        for i, t in enumerate(tiles, 1):
            tag = f"{t['X_tile']}X{t['Y_tile']}Y"
            pred_key = prediction_key(t["year"], t["X_tile"], t["Y_tile"])
            local_tif = os.path.join(args.output_dir, pred_key)
            print(f"[{i}/{len(tiles)}] {tag} year={t['year']}")
            print(f"  ARD keys to download:")
            for key in raw_ard_keys(t["year"], t["X_tile"], t["Y_tile"]):
                print(f"    s3://{args.bucket}/{key}")
            print(f"  Output: {local_tif}")
            print(f"  Upload: aws s3 cp {local_tif} s3://{args.bucket}/{pred_key}")
            print()
        return

    # Validate model path
    graph_file = os.path.join(args.model_path, "predict_graph-172.pb")
    if not os.path.isfile(graph_file):
        print(f"Model file not found: {graph_file}")
        print("Set --model-path to a directory containing predict_graph-172.pb")
        sys.exit(1)

    # Connect to S3
    print(f"Connecting to s3://{args.bucket} (region={args.region}) ...")
    store = _make_s3_store(args.bucket, args.region, args.aws_profile)
    validate_aws(store)
    print("  OK\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Lazy import — predict_tile pulls in TensorFlow which is slow
    from loaders.predict_tile import run as predict_run

    successes: list[dict] = []
    failures: list[dict] = []

    for i, t in enumerate(tiles, 1):
        tag = f"{t['X_tile']}X{t['Y_tile']}Y"
        print(f"[{i}/{len(tiles)}] {tag} year={t['year']}")
        t0 = time.time()

        try:
            # Check for locally available ARD
            all_keys = raw_ard_keys(t["year"], t["X_tile"], t["Y_tile"])
            local_keys = check_local_ard(args.output_dir, t["year"], t["X_tile"], t["Y_tile"])

            if len(local_keys) == len(all_keys):
                print("  ARD already present locally — skipping download")
            elif local_keys:
                missing_keys = set(all_keys) - set(local_keys)
                print(f"  {len(local_keys)}/{len(all_keys)} ARD files present locally, downloading {len(missing_keys)} remaining ...")
                download_ard_for_tile(store, args.output_dir, t["year"], t["X_tile"], t["Y_tile"], skip_existing=True)
            else:
                print("  Downloading ARD ...")
                download_ard_for_tile(store, args.output_dir, t["year"], t["X_tile"], t["Y_tile"])

            # Run prediction
            print("  Running inference ...")
            result = predict_run(
                year=t["year"],
                lon=t["lon"],
                lat=t["lat"],
                X_tile=t["X_tile"],
                Y_tile=t["Y_tile"],
                dest=args.output_dir,
                model_path=args.model_path,
            )

            elapsed = time.time() - t0

            if result["status"] == "success":
                pred_key = prediction_key(t["year"], t["X_tile"], t["Y_tile"])
                local_tif = os.path.join(args.output_dir, pred_key)
                print(f"  OK — {local_tif} ({elapsed:.0f}s)")
                successes.append({**t, "local_tif": local_tif, "pred_key": pred_key})
            else:
                print(f"  FAILED — {result.get('error_message', 'unknown error')}")
                failures.append({**t, "error": result.get("error_message", "unknown")})

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR — {e} ({elapsed:.0f}s)")
            traceback.print_exc()
            failures.append({**t, "error": str(e)})

        finally:
            if not args.keep_ard:
                cleanup_ard(args.output_dir, t["year"], t["X_tile"], t["Y_tile"])

        print()

    # Summary
    print("=" * 60)
    print(f"Results: {len(successes)} succeeded, {len(failures)} failed")
    print()

    if failures:
        print("Failed tiles:")
        for f in failures:
            print(f"  {f['X_tile']}X{f['Y_tile']}Y year={f['year']} — {f['error']}")
        print()

    if successes:
        print("Upload commands:")
        print()
        for s in successes:
            print(f"aws s3 cp {s['local_tif']} s3://{args.bucket}/{s['pred_key']}")
        print()

        # Also print a single combined command if multiple tiles
        if len(successes) > 1:
            tiles_dir = os.path.join(args.output_dir)
            print("Or sync the entire output directory:")
            # Find common year prefix
            years = sorted(set(s["year"] for s in successes))
            for year in years:
                local_tiles = os.path.join(args.output_dir, str(year), "tiles")
                print(f"aws s3 sync {local_tiles} s3://{args.bucket}/{year}/tiles/")
            print()


if __name__ == "__main__":
    main()
