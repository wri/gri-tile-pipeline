"""
Describe locally mirrored assets for a given tile and year.

Given a local root directory that mirrors the S3 prefix structure
(`year/x/y/...`), this script reports the dimensions of the expected
assets for a tile and, with `--verbose`, prints per-band statistics.

Example:
    python scripts/describe_assets.py --year 2022 --x 999 --y 988 \
        --local-root example --verbose

Only local files are inspected; no network calls are made.
"""

import argparse
import os
import numpy as np
import rasterio
import hickle as hkl


def format_shape(arr_shape):
    """Return a human-friendly shape string."""
    try:
        return "x".join(str(int(d)) for d in arr_shape)
    except Exception:
        return str(arr_shape)


def describe_tif(path, verbose=False):
    with rasterio.open(path) as ds:
        width, height = ds.width, ds.height
        count = ds.count
        crs = ds.crs
        dtype = ds.dtypes[0] if count > 0 else None
        print(
            f"  Type: GeoTIFF | Size: {width}x{height} | Bands: {count} | DType: {dtype} | CRS: {crs}"
        )
        if verbose and count > 0:
            # Print simple stats per band
            for b in range(1, count + 1):
                data = ds.read(b, masked=True)
                finite = np.asarray(data).astype(np.float64)
                finite = finite[np.isfinite(finite)]
                if finite.size == 0:
                    print(f"    Band {b}: no finite data")
                else:
                    print(
                        f"    Band {b}: min={finite.min():.4f} max={finite.max():.4f} "
                        f"mean={finite.mean():.4f} std={finite.std():.4f}"
                    )


def describe_npy(path, verbose=False):
    arr = np.load(path)
    print(f"  Type: NPY | Shape: {format_shape(arr.shape)} | DType: {arr.dtype}")
    if verbose:
        flat = arr.astype(np.float64).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            print("    No finite data")
        else:
            print(
                f"    Stats: min={flat.min():.4f} max={flat.max():.4f} "
                f"mean={flat.mean():.4f} std={flat.std():.4f}"
            )


def describe_hkl(path, verbose=False):
    obj = hkl.load(path)
    # Handle common container shapes
    if isinstance(obj, np.ndarray):
        print(
            f"  Type: HKL(np.ndarray) | Shape: {format_shape(obj.shape)} | DType: {obj.dtype}"
        )
        if verbose:
            flat = obj.astype(np.float64).ravel()
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                print("    No finite data")
            else:
                print(
                    f"    Stats: min={flat.min():.4f} max={flat.max():.4f} "
                    f"mean={flat.mean():.4f} std={flat.std():.4f}"
                )
    elif isinstance(obj, dict):
        keys = list(obj.keys())
        print(
            f"  Type: HKL(dict) | Keys: {len(keys)} -> {keys[:5]}{'...' if len(keys) > 5 else ''}"
        )
        if verbose:
            for k, v in obj.items():
                if isinstance(v, np.ndarray):
                    print(f"    {k}: shape={format_shape(v.shape)} dtype={v.dtype}")
                else:
                    t = type(v).__name__
                    print(f"    {k}: type={t}")
    else:
        t = type(obj).__name__
        print(f"  Type: HKL({t})")


def main():
    parser = argparse.ArgumentParser(
        description="Describe locally mirrored assets for a tile"
    )
    parser.add_argument(
        "-year", "--year", type=int, required=True, help="Year for the assets"
    )
    parser.add_argument("-x", "--x", type=int, required=True, help="X coordinate")
    parser.add_argument("-y", "--y", type=int, required=True, help="Y coordinate")
    parser.add_argument(
        "--local-root",
        type=str,
        required=True,
        help="Local root directory that mirrors the S3 prefix structure (e.g., example)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print per-band statistics"
    )

    args = parser.parse_args()

    # Expected asset paths relative to local root
    file_patterns = [
        f"{args.year}/composite/{args.x}/{args.y}/{args.x}X{args.y}Y.tif",
        f"{args.year}/ard/{args.x}/{args.y}/{args.x}X{args.y}Y_ard.hkl",
        f"{args.year}/tiles/{args.x}/{args.y}/{args.x}X{args.y}Y_FINAL.tif",
        f"{args.year}/change/{args.x}/{args.y}/{args.x}X{args.y}Y_ard.zip",
        f"{args.year}/processed/{args.x}/{args.y}/processed/0/0.npy",
        f"{args.year}/raw/{args.x}/{args.y}/raw/clouds/clean_steps_{args.x}X{args.y}Y.hkl",
        f"{args.year}/raw/{args.x}/{args.y}/raw/clouds/cloudmask_{args.x}X{args.y}Y.hkl",
        f"{args.year}/raw/{args.x}/{args.y}/raw/clouds/clouds_{args.x}X{args.y}Y.hkl",
        f"{args.year}/raw/{args.x}/{args.y}/raw/misc/dem_{args.x}X{args.y}Y.hkl",
        f"{args.year}/raw/{args.x}/{args.y}/raw/misc/s1_dates_{args.x}X{args.y}Y.hkl",
        f"{args.year}/raw/{args.x}/{args.y}/raw/misc/s2_dates_{args.x}X{args.y}Y.hkl",
        f"{args.year}/raw/{args.x}/{args.y}/raw/s1/{args.x}X{args.y}Y.hkl",
        f"{args.year}/raw/{args.x}/{args.y}/raw/s2_10/{args.x}X{args.y}Y.hkl",
        f"{args.year}/raw/{args.x}/{args.y}/raw/s2_20/{args.x}X{args.y}Y.hkl",
    ]

    print(f"Describing assets for year={args.year}, x={args.x}, y={args.y}")
    print("=" * 60)

    for rel_path in file_patterns:
        path = os.path.join(args.local_root, rel_path)
        if not os.path.exists(path):
            print(f"✗ MISSING: {rel_path}")
            continue

        print(f"✓ EXISTS: {rel_path}")
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".tif", ".tiff"):
                describe_tif(path, verbose=args.verbose)
            elif ext == ".npy":
                describe_npy(path, verbose=args.verbose)
            elif ext == ".hkl":
                describe_hkl(path, verbose=args.verbose)
            else:
                size = os.path.getsize(path)
                print(f"  Type: {ext or 'unknown'} | Size: {size} bytes (no parser)")
        except Exception as e:
            print(f"  ! Failed to describe: {e}")
        print()

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    exit(main())
