"""S1 → 10m target-grid alignment in predict_tile (_align_s1_to_grid)."""

import numpy as np
import pytest

from gri_tile_loaders.predict_tile import _align_s1_to_grid


def _s1(h: int, w: int, t: int = 12, c: int = 2) -> np.ndarray:
    """Strictly positive S1 stack whose values encode (row, col) so alignment is traceable."""
    rows = np.arange(h, dtype=np.float32)[:, None] * 1000
    cols = np.arange(w, dtype=np.float32)[None, :]
    plane = rows + cols + 1
    return np.broadcast_to(plane[None, :, :, None], (t, h, w, c)).copy()


@pytest.mark.parametrize(
    "h, w, H, W",
    [
        (156, 155, 620, 616),  # golden 1000X798Y / 1000X800Y: 4x upsample overshoots both dims
        (155, 155, 620, 616),  # golden 1000X799Y: overshoots width only
        (156, 155, 622, 616),  # golden 1000X800Y target
        (154, 154, 620, 616),  # 4x upsample undershoots both dims
        (155, 154, 620, 616),  # exact fit after upsample
    ],
)
def test_output_matches_target_with_no_zeroed_pixels(h, w, H, W):
    out = _align_s1_to_grid(_s1(h, w), H, W)
    assert out.shape == (12, H, W, 2)
    # Any zero here is a pixel the alignment dropped and back-filled with
    # constant 0 — the failure mode where a negative crop emptied the width axis.
    assert (out != 0).all()


def test_width_overshoot_is_center_cropped_not_emptied():
    # 155 px → 620 px after 4x repeat, 4 px wider than the 616 target.
    out = _align_s1_to_grid(_s1(155, 155), 620, 616)
    # Centered crop drops 2 px per side: target column 0 is native column 0
    # (its 4-px footprint starts at upsampled column 0, so column 2 is still native 0),
    # and the last target column is native column 154.
    row0 = out[0, 0, :, 0]
    assert row0[0] == 1  # native (0, 0)
    assert row0[-1] == 155  # native (0, 154)
    np.testing.assert_array_equal(np.unique(row0), np.arange(1, 156, dtype=np.float32))


def test_each_native_pixel_covers_its_4x4_footprint():
    out = _align_s1_to_grid(_s1(155, 154), 620, 616)
    native = _s1(155, 154)
    np.testing.assert_array_equal(out[:, 4 * 10:4 * 11, 4 * 20:4 * 21, :],
                                  np.broadcast_to(native[:, 10:11, 20:21, :], (12, 4, 4, 2)))


def test_already_on_grid_is_unchanged():
    s1 = _s1(620, 616)
    np.testing.assert_array_equal(_align_s1_to_grid(s1, 620, 616), s1)
