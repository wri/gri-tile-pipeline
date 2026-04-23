"""Parity test: feature stack from real ARD has correct shape and channel ranges."""

import os
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARD_DIR = REPO_ROOT / "example" / "raw_v2"

has_ard = pytest.mark.skipif(
    not ARD_DIR.is_dir(), reason=f"ARD directory not found: {ARD_DIR}"
)

pytestmark = [pytest.mark.parity, pytest.mark.slow, has_ard]


def test_feature_stack_shape_and_channels(ard_dir):
    """Build feature stack from real ARD and validate shape + channel ranges."""
    import hickle as hkl
    from skimage.transform import resize as sk_resize

    from gri_tile_pipeline.inference.normalize import MAX_ALL, MIN_ALL
    from gri_tile_pipeline.preprocessing.cloud_removal import (
        id_missing_px,
        interpolate_na_vals,
    )
    from gri_tile_pipeline.preprocessing.indices import make_indices
    from gri_tile_pipeline.preprocessing.temporal_resampling import resample_to_biweekly
    from gri_tile_pipeline.preprocessing.whittaker import WhittakerSmoother

    # Load raw data
    s2_10 = hkl.load(f"{ard_dir}/s2_10/1000X871Y.hkl").astype(np.float32) / 65535.0
    s2_20 = hkl.load(f"{ard_dir}/s2_20/1000X871Y.hkl").astype(np.float32) / 65535.0
    s1 = hkl.load(f"{ard_dir}/s1/1000X871Y.hkl").astype(np.float32) / 65535.0
    dem = hkl.load(f"{ard_dir}/misc/dem_1000X871Y.hkl").astype(np.float32) / 90.0
    s2_dates = hkl.load(f"{ard_dir}/misc/s2_dates_1000X871Y.hkl")

    T, H, W = s2_10.shape[:3]

    # Upsample 20m
    s2_20_up = np.zeros((T, H, W, s2_20.shape[-1]), dtype=np.float32)
    for t in range(T):
        for b in range(s2_20.shape[-1]):
            s2_20_up[t, :, :, b] = sk_resize(
                s2_20[t, :, :, b], (H, W), order=1, preserve_range=True
            )
    s2_full = np.concatenate([s2_10, s2_20_up], axis=-1)

    # Cleanup
    missing = id_missing_px(s2_full)
    if len(missing) > 0:
        s2_full = np.delete(s2_full, missing, axis=0)
    s2_full = interpolate_na_vals(s2_full)
    s2_full = np.clip(s2_full, 0, 1)

    # Temporal resampling
    s2_24, _ = resample_to_biweekly(s2_full, s2_dates)

    # Indices
    indices_raw = make_indices(s2_24)

    # Whittaker
    smoother = WhittakerSmoother(
        lmbd=100.0, size=24, nbands=10, dimx=H, dimy=W, outsize=12
    )
    s2_12 = smoother.interpolate_array(s2_24)

    sm_idx = WhittakerSmoother(
        lmbd=100.0, size=24, nbands=4, dimx=H, dimy=W, outsize=12
    )
    idx_12 = sm_idx.interpolate_array(indices_raw)

    # S1 resize
    s1_r = np.zeros((min(s1.shape[0], 12), H, W, 2), dtype=np.float32)
    for t in range(s1_r.shape[0]):
        for b in range(2):
            s1_r[t, :, :, b] = sk_resize(
                s1[t, :, :, b], (H, W), order=1, preserve_range=True
            )
    if s1_r.shape[0] < 12:
        pad_t = 12 - s1_r.shape[0]
        s1_r = np.pad(s1_r, ((0, pad_t), (0, 0), (0, 0), (0, 0)), "edge")

    # Quarterly reduction
    s2_q = np.median(s2_12.reshape(4, 3, H, W, 10), axis=1)
    idx_q = np.median(idx_12.reshape(4, 3, H, W, 4), axis=1)
    s1_q = np.median(s1_r.reshape(4, 3, H, W, 2), axis=1)

    # Build feature stack
    feature = np.zeros((5, H, W, 17), dtype=np.float32)
    feature[:4, :, :, :10] = s2_q
    feature[:4, :, :, 11:13] = s1_q
    feature[:4, :, :, 13:] = idx_q
    feature[:, :, :, 10] = np.broadcast_to(dem, (5, H, W))
    feature[-1, :, :, :10] = np.median(s2_12, axis=0)
    feature[-1, :, :, 11:13] = np.median(s1_r, axis=0)
    feature[-1, :, :, 13:] = np.median(idx_12, axis=0)

    # Assertions
    assert feature.shape == (5, H, W, 17), f"Expected (5, {H}, {W}, 17), got {feature.shape}"
    assert feature.dtype == np.float32

    # All 17 channels should be populated (non-zero)
    for ch in range(17):
        ch_max = feature[:, :, :, ch].max()
        assert ch_max > 0, f"Channel {ch} is all zeros"

    # DEM should be non-negative
    assert feature[:, :, :, 10].min() >= 0, "DEM channel has negative values"

    # S1 channels should be in [0, 1]
    for ch in [11, 12]:
        assert feature[:, :, :, ch].min() >= -0.01, f"S1 channel {ch} has values < 0"
        assert feature[:, :, :, ch].max() <= 1.01, f"S1 channel {ch} has values > 1"

    # S2 channels (0-9) should be in [0, 1]
    for ch in range(10):
        assert feature[:, :, :, ch].min() >= -0.01, f"S2 channel {ch} < 0"
        assert feature[:, :, :, ch].max() <= 1.5, f"S2 channel {ch} > 1.5"

    # Quarterly shape correct
    assert s2_q.shape == (4, H, W, 10)
    assert idx_q.shape == (4, H, W, 4)
    assert s1_q.shape == (4, H, W, 2)
