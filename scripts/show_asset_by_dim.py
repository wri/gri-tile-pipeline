"""
Visualize a local asset for a given dimension.

Given a filepath (`year/x/y/...`), this script shows plots representing the specified 
dimension of the target file. 

Example: 
    python scripts/show_asset_by_dim.py --filepath example/2022/ard/999/988/999X988Y_ard.hkl --dim 2

"""

import argparse
import math
import os
from typing import Optional

import hickle as hkl
import matplotlib.pyplot as plt
import numpy as np
import rasterio


def compute_global_min_max(panels: list[np.ndarray]) -> tuple[float, float]:
    global_min = None
    global_max = None
    for panel in panels:
        arr = np.asarray(panel, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        mn = float(finite.min())
        mx = float(finite.max())
        global_min = mn if global_min is None else min(global_min, mn)
        global_max = mx if global_max is None else max(global_max, mx)
    if global_min is None or global_max is None:
        # Fallback to zeros if everything is non-finite
        return 0.0, 1.0
    if global_min == global_max:
        # Avoid zero range for imshow
        return global_min, global_min + 1e-9
    return global_min, global_max


def two_largest_axes(shape: tuple[int, ...]) -> list[int]:
    if len(shape) <= 2:
        return list(range(len(shape)))
    # pick indices of the two largest dimensions; preserve sorted order
    largest = np.argsort(shape)[-2:]
    return sorted(largest.tolist())


def reduce_to_2d_preserve_spatial(arr: np.ndarray, spatial_axes: list[int]) -> np.ndarray:
    # Reduce all non-spatial axes by mean to obtain a 2D array in spatial axes order
    if arr.ndim <= 2:
        return arr
    # Bring spatial axes to the end in order (H, W)
    other_axes = [ax for ax in range(arr.ndim) if ax not in spatial_axes]
    perm = other_axes + spatial_axes
    trans = np.transpose(arr, axes=perm)
    # Reduce all leading non-spatial dims by mean
    while trans.ndim > 2:
        trans = trans.mean(axis=0)
    return trans


def plot_panels(panels: list[np.ndarray], titles: list[str], cmap: str, percentile: float):
    n = len(panels)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    vmin, vmax = compute_global_min_max(panels)
    for i in range(rows * cols):
        ax = axes.flat[i]
        if i < n:
            img = np.asarray(panels[i], dtype=np.float64)
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(titles[i])
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_histograms(panels: list[np.ndarray], titles: list[str], bins: int = 50):
    n = len(panels)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharey=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    vmin, vmax = compute_global_min_max(panels)

    # Precompute global max count for consistent y-axis across histograms
    max_count = 0
    finite_arrays: list[np.ndarray] = []
    for i in range(n):
        img = np.asarray(panels[i], dtype=np.float64).ravel()
        finite = img[np.isfinite(img)]
        finite_arrays.append(finite)
        if finite.size > 0:
            counts, _ = np.histogram(finite, bins=bins, range=(vmin, vmax))
            local_max = int(counts.max()) if counts.size > 0 else 0
            if local_max > max_count:
                max_count = local_max

    for i in range(rows * cols):
        ax = axes.flat[i]
        if i < n:
            finite = finite_arrays[i]
            if finite.size == 0:
                ax.text(0.5, 0.5, 'No finite data', ha='center', va='center')
                ax.axis('off')
            else:
                ax.hist(finite, bins=bins, range=(vmin, vmax), color='gray', alpha=0.7)
                ax.set_title(f"{titles[i]} histogram")
                ax.grid(True, alpha=0.2)
                ax.set_ylim(0, max_count if max_count > 0 else 1)
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()


def load_tif_panels(path: str, max_panels: int) -> tuple[list[np.ndarray], list[str]]:
    panels, titles = [], []
    with rasterio.open(path) as ds:
        count = ds.count
        for b in range(1, count + 1):
            data = ds.read(b, masked=True)
            panels.append(np.asarray(data))
            titles.append(f"Band {b}")
            if len(panels) >= max_panels:
                break
    return panels, titles


def load_npy_panels(path: str, dim: int, max_panels: int) -> tuple[list[np.ndarray], list[str]]:
    arr = np.load(path)
    return load_npy_panels_from_array(arr, dim=dim, max_panels=max_panels)


def load_hkl_panels(path: str, dim: int, key: Optional[str], max_panels: int) -> tuple[list[np.ndarray], list[str]]:
    obj = hkl.load(path)
    if isinstance(obj, dict):
        if not key or key not in obj:
            raise ValueError(f"HKL(dict) requires --key from {list(obj.keys())}")
        obj = obj[key]
    if not isinstance(obj, np.ndarray):
        raise TypeError(f"HKL content is {type(obj).__name__}, not ndarray")
    return load_npy_panels_from_array(obj, dim=dim, max_panels=max_panels)


def load_npy_panels_from_array(arr: np.ndarray, dim: int, max_panels: int) -> tuple[list[np.ndarray], list[str]]:
    panels, titles = [], []
    if arr.ndim <= 2:
        panels.append(arr)
        titles.append("Array")
        return panels, titles
    if dim < 0 or dim >= arr.ndim:
        dim = 0
    spatial_axes = two_largest_axes(arr.shape)
    # If user picked a spatial axis, prefer a non-spatial one
    if dim in spatial_axes:
        non_spatial = [ax for ax in range(arr.ndim) if ax not in spatial_axes]
        if non_spatial:
            dim = non_spatial[0]
    slices = arr.shape[dim]
    # pick up to max_panels indices evenly spaced
    if max_panels >= slices:
        indices = list(range(slices))
    else:
        indices = np.linspace(0, slices - 1, num=max_panels, dtype=int).tolist()
    for i in indices:
        slicer = [slice(None)] * arr.ndim
        slicer[dim] = i
        sliced = arr[tuple(slicer)]
        img2 = reduce_to_2d_preserve_spatial(sliced, spatial_axes=[ax - (1 if ax > dim else 0) for ax in spatial_axes if ax != dim])
        panels.append(img2)
        titles.append(f"{dim=} idx={i}")
    return panels, titles


def main():
    parser = argparse.ArgumentParser(description='Visualize a local asset by dimension/band')
    parser.add_argument('--filepath', type=str, required=True, help='Path to local file (e.g., example/2022/...)')
    parser.add_argument('--dim', type=int, default=0, help='Axis to visualize for arrays (ignored for GeoTIFF; bands are shown)')
    parser.add_argument('--key', type=str, default=None, help='Optional key for HKL dicts to select an array')
    parser.add_argument('--max-panels', type=int, default=6, help='Maximum number of panels to display')
    parser.add_argument('--percentile', type=float, default=2.0, help='Percentile stretch for visualization (0 disables)')
    parser.add_argument('--cmap', type=str, default='viridis', help='Matplotlib colormap')
    parser.add_argument('--hist', action='store_true', help='Also display a histogram for each panel')

    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"File not found: {args.filepath}")
        return 1

    ext = os.path.splitext(args.filepath)[1].lower()
    try:
        if ext in ('.tif', '.tiff'):
            panels, titles = load_tif_panels(args.filepath, args.max_panels)
        elif ext == '.npy':
            panels, titles = load_npy_panels(args.filepath, args.dim, args.max_panels)
        elif ext == '.hkl':
            panels, titles = load_hkl_panels(args.filepath, args.dim, args.key, args.max_panels)
        else:
            print(f"Unsupported file extension: {ext}")
            return 1
    except Exception as e:
        print(f"Failed to prepare panels: {e}")
        return 1

    if not panels:
        print("No panels to display")
        return 1

    try:
        plot_panels(panels, titles, cmap=args.cmap, percentile=args.percentile)
        if args.hist:
            plot_histograms(panels, titles)
    except Exception as e:
        print(f"Failed to render panels: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
