"""Shared utilities for loader Lambda workers.

Kept in `loaders/` (rather than `src/`) so Lambda deployments stay
self-contained without needing the full `gri_tile_pipeline` package.
"""

from __future__ import annotations

import os
import tempfile

import hickle as hkl
import numpy as np
import obstore as obs
from loguru import logger


def make_bbox(initial_bbx: list, expansion: int = 10) -> list:
    """Expand a point bbox by *expansion* units of ~300 m (1/360 degree each)."""
    multiplier = 1 / 360
    bbx = initial_bbx.copy()
    bbx[0] -= expansion * multiplier
    bbx[1] -= expansion * multiplier
    bbx[2] += expansion * multiplier
    bbx[3] += expansion * multiplier
    return bbx


def obstore_put_hkl(store, relpath: str, obj) -> None:
    """Serialize *obj* with hickle (gzip) and write to *store* at *relpath*."""
    tmp = tempfile.NamedTemporaryFile(suffix=".hkl", delete=False)
    tmp.close()
    try:
        hkl.dump(obj, tmp.name, mode="w", compression="gzip")
        with open(tmp.name, "rb") as f:
            data = f.read()
        try:
            obs.put(store, relpath, data)
        except Exception:
            logger.exception(f"Failed to write object store key: {relpath}")
            raise
        logger.debug(f"Saved {relpath} ({len(data) / 1024:.1f} KB)")
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass


def compute_band_stats(arr: np.ndarray) -> dict:
    """Compute summary stats ignoring zeros as nodata.

    Returns dict with min, max, mean, std, p5, p50, p95, valid_ratio, count, valid_count.
    """
    vals = arr.astype(np.float32)
    total = int(vals.size)
    mask = vals > 0
    valid = vals[mask]
    if valid.size == 0:
        return {
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "std": 0.0,
            "p5": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "valid_ratio": 0.0,
            "count": total,
            "valid_count": 0,
        }
    p5, p50, p95 = np.percentile(valid, [5, 50, 95])
    return {
        "min": int(valid.min()),
        "max": int(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "p5": float(p5),
        "p50": float(p50),
        "p95": float(p95),
        "valid_ratio": float(valid.size / total) if total > 0 else 0.0,
        "count": total,
        "valid_count": int(valid.size),
    }
