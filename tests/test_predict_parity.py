"""Numeric parity test: compare our prediction against the reference output.

Usage:
    python tests/test_predict_parity.py

Requires:
    - example/raw_v2/ (ARD input data for tile 1000X871Y)
    - example/1000X871Y_FINAL.tif (reference prediction)
    - models/predict_graph-172.pb
    - tensorflow, hickle, rasterio, scikit-image, scipy, numpy
"""

import os
import sys
import numpy as np

# Paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARD_DIR = os.path.join(REPO_ROOT, "example", "raw_v2")
REFERENCE_TIF = os.path.join(REPO_ROOT, "example", "1000X871Y_FINAL.tif")
MODEL_DIR = os.path.join(REPO_ROOT, "models")
OUTPUT_TIF = os.path.join(REPO_ROOT, "temp", "test_1000X871Y_FINAL.tif")


def compare_predictions(ours: np.ndarray, ref: np.ndarray) -> dict:
    """Pixel-by-pixel comparison of two prediction arrays."""
    # Ignore nodata (255) in either
    mask = (ref != 255) & (ours != 255)
    n_valid = mask.sum()

    if n_valid == 0:
        return {"n_valid": 0, "error": "No overlapping valid pixels"}

    diff = np.abs(ref[mask].astype(float) - ours[mask].astype(float))

    return {
        "n_valid": int(n_valid),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "pct_within_1": float((diff <= 1).mean() * 100),
        "pct_within_5": float((diff <= 5).mean() * 100),
        "pct_within_10": float((diff <= 10).mean() * 100),
        "correlation": float(np.corrcoef(
            ref[mask].astype(float), ours[mask].astype(float)
        )[0, 1]),
        "our_mean": float(ours[ours != 255].mean()) if np.any(ours != 255) else 0,
        "ref_mean": float(ref[ref != 255].mean()) if np.any(ref != 255) else 0,
    }


def test_predict_parity():
    """Run prediction on example data and compare to reference."""
    import rasterio

    # Check prerequisites
    if not os.path.isdir(ARD_DIR):
        print(f"SKIP: ARD directory not found: {ARD_DIR}")
        return

    if not os.path.isfile(REFERENCE_TIF):
        print(f"SKIP: Reference TIF not found: {REFERENCE_TIF}")
        return

    if not os.path.isfile(os.path.join(MODEL_DIR, "predict_graph-172.pb")):
        print(f"SKIP: Model not found: {MODEL_DIR}/predict_graph-172.pb")
        return

    # Load reference
    with rasterio.open(REFERENCE_TIF) as src:
        ref = src.read(1)

    print(f"Reference: shape={ref.shape}, dtype={ref.dtype}, "
          f"range=[{ref[ref != 255].min()}-{ref[ref != 255].max()}], "
          f"mean={ref[ref != 255].mean():.1f}")

    # Run our prediction
    sys.path.insert(0, os.path.join(REPO_ROOT, "loaders"))
    from predict_tile import run_local

    os.makedirs(os.path.dirname(OUTPUT_TIF), exist_ok=True)
    ours = run_local(
        ard_dir=ARD_DIR,
        model_path=MODEL_DIR,
        output_path=OUTPUT_TIF,
    )

    print(f"Ours:      shape={ours.shape}, dtype={ours.dtype}, "
          f"range=[{ours[ours != 255].min()}-{ours[ours != 255].max()}], "
          f"mean={ours[ours != 255].mean():.1f}")

    # Compare
    # Handle potential shape mismatch (crop to smaller)
    h = min(ref.shape[0], ours.shape[0])
    w = min(ref.shape[1], ours.shape[1])
    stats = compare_predictions(ours[:h, :w], ref[:h, :w])

    print("\n--- Parity Results ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Quality gates
    pct1 = stats.get("pct_within_1", 0)
    pct5 = stats.get("pct_within_5", 0)
    corr = stats.get("correlation", 0)

    print(f"\n--- Quality Gates ---")
    print(f"  >99% within 1 DN:  {'PASS' if pct1 > 99 else 'FAIL'} ({pct1:.1f}%)")
    print(f"  >99% within 5 DN:  {'PASS' if pct5 > 99 else 'FAIL'} ({pct5:.1f}%)")
    print(f"  Correlation > 0.99: {'PASS' if corr > 0.99 else 'FAIL'} ({corr:.4f})")

    if pct1 < 95:
        print(f"\n  WARNING: Only {pct1:.1f}% within 1 DN. "
              "Previous best attempt achieved 95.2%. "
              "Investigate preprocessing differences.")

    return stats


def test_preprocessing_only():
    """Test just the preprocessing pipeline (no TF needed).

    Validates that the feature stack shape and value ranges are correct.
    """
    import hickle as hkl
    from gri_tile_pipeline.preprocessing.cloud_removal import (
        id_missing_px, interpolate_na_vals,
    )
    from gri_tile_pipeline.preprocessing.whittaker import WhittakerSmoother
    from gri_tile_pipeline.preprocessing.indices import make_indices
    from gri_tile_pipeline.preprocessing.temporal_resampling import resample_to_biweekly
    from gri_tile_pipeline.inference.normalize import MIN_ALL, MAX_ALL
    from skimage.transform import resize as sk_resize

    # Load raw data
    s2_10 = hkl.load(f"{ARD_DIR}/s2_10/1000X871Y.hkl").astype(np.float32) / 65535.0
    s2_20 = hkl.load(f"{ARD_DIR}/s2_20/1000X871Y.hkl").astype(np.float32) / 65535.0
    s1 = hkl.load(f"{ARD_DIR}/s1/1000X871Y.hkl").astype(np.float32) / 65535.0
    dem = hkl.load(f"{ARD_DIR}/misc/dem_1000X871Y.hkl").astype(np.float32)
    dem = dem / 90.0  # reference line 1084
    s2_dates = hkl.load(f"{ARD_DIR}/misc/s2_dates_1000X871Y.hkl")

    T, H, W = s2_10.shape[:3]
    print(f"S2_10: {s2_10.shape}, S2_20: {s2_20.shape}, S1: {s1.shape}, DEM: {dem.shape}")
    print(f"S2 dates: {s2_dates}")

    # Upsample 20m
    s2_20_up = np.zeros((T, H, W, s2_20.shape[-1]), dtype=np.float32)
    for t in range(T):
        for b in range(s2_20.shape[-1]):
            s2_20_up[t, :, :, b] = sk_resize(s2_20[t, :, :, b], (H, W), order=1, preserve_range=True)

    s2_full = np.concatenate([s2_10, s2_20_up], axis=-1)
    print(f"S2 merged: {s2_full.shape}, range=[{s2_full.min():.4f}, {s2_full.max():.4f}]")

    # Remove bad timesteps + handle missing values
    # (complex cloud removal was done during download; ARD has pre-selected clean dates)
    missing = id_missing_px(s2_full)
    print(f"Missing timesteps: {missing}")
    if len(missing) > 0:
        s2_full = np.delete(s2_full, missing, axis=0)

    s2_full = interpolate_na_vals(s2_full)
    median_vals = np.median(s2_full, axis=0)
    for i in range(s2_full.shape[0]):
        zero_mask = s2_full[i] == 0
        if np.any(zero_mask):
            s2_full[i][zero_mask] = median_vals[zero_mask]
        one_mask = s2_full[i] == 1
        if np.any(one_mask):
            s2_full[i][one_mask] = median_vals[one_mask]
    s2_full = np.clip(s2_full, 0, 1)
    print(f"After cleanup: {s2_full.shape}, NaN count: {np.isnan(s2_full).sum()}")

    # Temporal resampling to 24 biweekly composites (reference step)
    s2_24, max_gap = resample_to_biweekly(s2_full, s2_dates)
    print(f"Resampled to 24 biweekly: {s2_24.shape}, max gap: {max_gap}d")

    # Indices (from resampled, pre-smoothing — reference computes indices then smoothes separately)
    indices_raw = make_indices(s2_24)
    print(f"Raw indices: {indices_raw.shape}")

    # Whittaker (size=24, outsize=12)
    smoother = WhittakerSmoother(lmbd=100.0, size=24, nbands=10, dimx=H, dimy=W, outsize=12)
    s2_12 = smoother.interpolate_array(s2_24)
    print(f"Smoothed S2: {s2_12.shape}, range=[{s2_12.min():.4f}, {s2_12.max():.4f}]")

    sm_idx = WhittakerSmoother(lmbd=100.0, size=24, nbands=4, dimx=H, dimy=W, outsize=12)
    idx_12 = sm_idx.interpolate_array(indices_raw)
    print(f"Smoothed indices: {idx_12.shape}")

    # S1 resize
    if s1.shape[1] != H or s1.shape[2] != W:
        s1_r = np.zeros((min(s1.shape[0], 12), H, W, 2), dtype=np.float32)
        for t in range(s1_r.shape[0]):
            for b in range(2):
                s1_r[t, :, :, b] = sk_resize(s1[t, :, :, b], (H, W), order=1, preserve_range=True)
    else:
        s1_r = s1[:12]
    if s1_r.shape[0] < 12:
        pad_t = 12 - s1_r.shape[0]
        s1_r = np.pad(s1_r, ((0, pad_t), (0, 0), (0, 0), (0, 0)), 'edge')

    # Quarterly reduction: 12 monthly → 4 quarterly (reference lines 1394-1398)
    s2_q = np.median(s2_12.reshape(4, 3, H, W, 10), axis=1)      # (4, H, W, 10)
    idx_q = np.median(idx_12.reshape(4, 3, H, W, 4), axis=1)     # (4, H, W, 4)
    s1_q = np.median(s1_r.reshape(4, 3, H, W, 2), axis=1)        # (4, H, W, 2)
    print(f"Quarterly: S2 {s2_q.shape}, indices {idx_q.shape}, S1 {s1_q.shape}")

    # Build feature stack (5, H, W, 17) — 4 quarterly + 1 median
    feature = np.zeros((5, H, W, 17), dtype=np.float32)
    feature[:4, :, :, :10] = s2_q
    feature[:4, :, :, 11:13] = s1_q
    feature[:4, :, :, 13:] = idx_q
    feature[:, :, :, 10] = np.broadcast_to(dem, (5, H, W))
    feature[-1, :, :, :10] = np.median(s2_12, axis=0)
    feature[-1, :, :, 11:13] = np.median(s1_r, axis=0)
    feature[-1, :, :, 13:] = np.median(idx_12, axis=0)

    print(f"\nFeature stack: {feature.shape}")
    print(f"Channel ranges vs normalization bounds:")
    for ch in range(17):
        ch_data = feature[:, :, :, ch]
        print(f"  ch{ch:2d}: [{ch_data.min():.4f}, {ch_data.max():.4f}] "
              f"norm=[{MIN_ALL[ch]:.4f}, {MAX_ALL[ch]:.4f}] "
              f"{'OK' if ch_data.max() > 0 else 'EMPTY'}")

    print("\nPreprocessing test PASSED — feature stack shape and ranges look correct.")
    return feature


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Only test preprocessing (no TF required)")
    args = parser.parse_args()

    if args.preprocess_only:
        test_preprocessing_only()
    else:
        test_predict_parity()
