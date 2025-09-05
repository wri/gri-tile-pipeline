"""
Generate descriptive statistics for each band of a local asset for a given dimension.

Given a filepath (`year/x/y/...`), this script reports the dimensions and band statistics of the target file.

Example:
    python scripts/describe_asset_by_dim.py --filepath example/2022/ard/999/988/999X988Y_ard.hkl --dim 2

"""

import argparse
import os
import numpy as np
import rasterio
import hickle as hkl


def summarize_array(arr: np.ndarray, axis: int):
    print(f"Array shape: {arr.shape}, dtype: {arr.dtype}")
    if axis < 0 or axis >= arr.ndim:
        print(f"Requested axis {axis} out of bounds for array with {arr.ndim} dims")
        return
    num_slices = arr.shape[axis]
    print(f"Summarizing along axis {axis} with {num_slices} slices")

    for idx in range(num_slices):
        slicer = [slice(None)] * arr.ndim
        slicer[axis] = idx
        data = arr[tuple(slicer)].astype(np.float64)
        data = data[np.isfinite(data)]
        if data.size == 0:
            print(f"  Slice {idx}: no finite data")
        else:
            print(
                f"  Slice {idx}: min={data.min():.4f} max={data.max():.4f} "
                f"mean={data.mean():.4f} std={data.std():.4f}"
            )


def describe_tif(path: str):
    with rasterio.open(path) as ds:
        width, height = ds.width, ds.height
        count = ds.count
        dtype = ds.dtypes[0] if count > 0 else None
        print(f"GeoTIFF {path}")
        print(
            f"  Size: {width}x{height} | Bands: {count} | DType: {dtype} | CRS: {ds.crs}"
        )
        for b in range(1, count + 1):
            data = ds.read(b, masked=True)
            finite = np.asarray(data).astype(np.float64)
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                print(f"  Band {b}: no finite data")
            else:
                print(
                    f"  Band {b}: min={finite.min():.4f} max={finite.max():.4f} "
                    f"mean={finite.mean():.4f} std={finite.std():.4f}"
                )


def describe_npy(path: str, dim: int):
    arr = np.load(path)
    print(f"NPY {path}")
    summarize_array(arr, dim)


def describe_hkl(path: str, dim: int, key: str | None):
    obj = hkl.load(path)
    if isinstance(obj, np.ndarray):
        print(f"HKL(np.ndarray) {path}")
        summarize_array(obj, dim)
    elif isinstance(obj, dict):
        if key is None:
            print("HKL(dict) requires --key to select an array entry. Available keys:")
            print("  ", list(obj.keys()))
            return
        if key not in obj:
            print(
                f"Key '{key}' not found in HKL(dict). Available keys: {list(obj.keys())}"
            )
            return
        val = obj[key]
        if isinstance(val, np.ndarray):
            print(f"HKL(dict['{key}'] np.ndarray) {path}")
            summarize_array(val, dim)
        else:
            print(f"HKL(dict['{key}']) is type {type(val).__name__}, not an array")
    else:
        print(
            f"HKL object type {type(obj).__name__} not supported for dim-wise summary"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Describe a local asset by dimension/band"
    )
    parser.add_argument(
        "--filepath",
        type=str,
        required=True,
        help="Path to local file (e.g., example/2022/...)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=0,
        help="Axis to summarize for arrays (ignored for GeoTIFF; bands are used)",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Optional key for HKL dicts to select an array",
    )

    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"File not found: {args.filepath}")
        return 1

    ext = os.path.splitext(args.filepath)[1].lower()
    try:
        if ext in (".tif", ".tiff"):
            describe_tif(args.filepath)
        elif ext == ".npy":
            describe_npy(args.filepath, args.dim)
        elif ext == ".hkl":
            describe_hkl(args.filepath, args.dim, args.key)
        else:
            print(f"Unsupported file extension: {ext}")
            return 1
    except Exception as e:
        print(f"Failed to describe file: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
