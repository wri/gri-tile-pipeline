"""Shared parity comparison metrics for prediction tests."""

from __future__ import annotations

import numpy as np


def compare_predictions(ours: np.ndarray, ref: np.ndarray) -> dict:
    """Pixel-by-pixel comparison of two prediction arrays.

    Both arrays are expected to be uint8 with 255 = nodata.
    Returns a dict of scalar metrics.
    """
    mask = (ref != 255) & (ours != 255)
    n_valid = int(mask.sum())

    if n_valid == 0:
        return {"n_valid": 0, "error": "No overlapping valid pixels"}

    r = ref[mask].astype(float)
    o = ours[mask].astype(float)
    diff = np.abs(r - o)

    # Correlation excluding outliers (>30 DN diff, typically cloud-affected)
    non_outlier = diff <= 30
    corr_excl = float("nan")
    pct_outlier = float((~non_outlier).mean() * 100)
    if non_outlier.sum() > 100:
        corr_excl = float(np.corrcoef(r[non_outlier], o[non_outlier])[0, 1])

    return {
        "n_valid": n_valid,
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "pct_within_1": float((diff <= 1).mean() * 100),
        "pct_within_5": float((diff <= 5).mean() * 100),
        "pct_within_10": float((diff <= 10).mean() * 100),
        "correlation": float(np.corrcoef(r, o)[0, 1]),
        "correlation_excl_outliers": corr_excl,
        "pct_outlier_30dn": pct_outlier,
        "our_mean": float(ours[ours != 255].mean()) if np.any(ours != 255) else 0,
        "ref_mean": float(ref[ref != 255].mean()) if np.any(ref != 255) else 0,
    }


def aggregate_golden_report(per_tile_stats: dict[str, dict]) -> dict:
    """Aggregate per-tile parity results into a summary."""
    tiles = list(per_tile_stats.keys())
    if not tiles:
        return {}

    def _get(key):
        return [per_tile_stats[t][key] for t in tiles if key in per_tile_stats[t]]

    pct1 = _get("pct_within_1")
    pct10 = _get("pct_within_10")
    corr = _get("correlation")
    corr_excl = [v for v in _get("correlation_excl_outliers") if not np.isnan(v)]

    return {
        "tiles_tested": len(tiles),
        "mean_pct_within_1": float(np.mean(pct1)) if pct1 else 0,
        "min_pct_within_1": float(np.min(pct1)) if pct1 else 0,
        "worst_tile_1dn": tiles[int(np.argmin(pct1))] if pct1 else "",
        "mean_pct_within_10": float(np.mean(pct10)) if pct10 else 0,
        "min_pct_within_10": float(np.min(pct10)) if pct10 else 0,
        "mean_correlation": float(np.mean(corr)) if corr else 0,
        "mean_correlation_excl_outliers": float(np.mean(corr_excl)) if corr_excl else 0,
    }


def compare_intermediates(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    """Compare two intermediate arrays and return divergence metrics.

    Args:
        name: Label for the comparison (e.g., "cloud_probs").
        a: First array.
        b: Second array (same shape expected).

    Returns:
        Dict with max_diff, mean_diff, pct_exact, pct_within_1e6.
    """
    if a.shape != b.shape:
        return {
            "name": name,
            "error": f"Shape mismatch: {a.shape} vs {b.shape}",
        }
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "name": name,
        "shape": a.shape,
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "pct_exact": float((diff == 0).mean() * 100),
        "pct_within_1e-6": float((diff <= 1e-6).mean() * 100),
    }


def print_parity_table(per_tile_stats: dict[str, dict]) -> None:
    """Print a formatted parity comparison table."""
    header = (
        f"{'Tile':<16} | {'MeanDiff':>8} | {'%<=1DN':>7} | {'%<=5DN':>7} "
        f"| {'%<=10DN':>7} | {'Corr':>7} | {'CorrExcl':>8} | {'OurMean':>7} | {'RefMean':>7}"
    )
    print("\n=== Golden Tile Parity Report ===")
    print(header)
    print("-" * len(header))

    for tile, s in sorted(per_tile_stats.items()):
        if "error" in s:
            print(f"{tile:<16} | ERROR: {s['error']}")
            continue
        print(
            f"{tile:<16} | {s['mean_abs_diff']:8.2f} | {s['pct_within_1']:6.1f}% "
            f"| {s['pct_within_5']:6.1f}% | {s['pct_within_10']:6.1f}% "
            f"| {s['correlation']:7.4f} | {s['correlation_excl_outliers']:8.4f} "
            f"| {s['our_mean']:7.1f} | {s['ref_mean']:7.1f}"
        )

    agg = aggregate_golden_report(per_tile_stats)
    if agg:
        print("-" * len(header))
        print(
            f"{'AGGREGATE':<16} |          | {agg['mean_pct_within_1']:6.1f}% "
            f"|         | {agg['mean_pct_within_10']:6.1f}% "
            f"| {agg['mean_correlation']:7.4f} | {agg['mean_correlation_excl_outliers']:8.4f} |"
        )
        print(f"  Worst tile (1DN): {agg['worst_tile_1dn']} ({agg['min_pct_within_1']:.1f}%)")
    print()
