"""DSen2 bilinear / CNN 20 m -> 10 m super-resolution.

Ported from reference ``download_and_predict_job.py`` lines 125-177.
Requires a TensorFlow super-resolution session to be passed in.
Falls back to bilinear-only if no session is available.
"""

from __future__ import annotations

import numpy as np
from loguru import logger


def superresolve_tile(
    arr: np.ndarray,
    sess=None,
    sr_logits=None,
    sr_inp=None,
    sr_inp_bilinear=None,
    window_size: int = 110,
) -> np.ndarray:
    """Super-resolve 20 m bands in a ``(T, H, W, 10)`` tile.

    Bands 0-3 are 10 m native; bands 4-9 have been bilinearly upsampled
    to 10 m resolution before calling this function.

    If a TF session is provided, applies CNN-based DSen2 super-resolution
    to the 20 m bands in overlapping windows.

    Args:
        arr: ``(T, H, W, 10)`` float32 array.
        sess: ``tf.compat.v1.Session`` for the super-resolution model.
        sr_logits: Output tensor of the SR model.
        sr_inp: Input tensor (full 10 bands).
        sr_inp_bilinear: Input tensor (bilinear 20 m bands only).
        window_size: Tile processing window size.

    Returns:
        ``(T, H, W, 10)`` array with super-resolved 20 m bands.
    """
    if sess is None:
        logger.debug("No super-resolution session — using bilinear upsampling only")
        return arr

    def _worker(chunk: np.ndarray) -> np.ndarray:
        padded = np.pad(chunk, ((0, 0), (4, 4), (4, 4), (0, 0)), "reflect")
        bilinear = padded[..., 4:]
        resolved = sess.run(
            [sr_logits],
            feed_dict={sr_inp: padded, sr_inp_bilinear: bilinear},
        )[0]
        resolved = resolved[:, 4:-4, 4:-4, :]
        chunk[..., 4:] = resolved
        return chunk

    step = window_size
    x_range = list(range(0, arr.shape[1] - window_size, step)) + [
        arr.shape[1] - window_size
    ]
    y_range = list(range(0, arr.shape[2] - window_size, step)) + [
        arr.shape[2] - window_size
    ]

    # Keep original border strips to avoid feeding partially-resolved input
    x_end = np.copy(arr[:, x_range[-1]:, ...])
    y_end = np.copy(arr[:, :, y_range[-1]:, ...])

    for x in x_range:
        for y in y_range:
            if x != x_range[-1] and y != y_range[-1]:
                chunk = arr[:, x : x + window_size, y : y + window_size, ...]
                arr[:, x : x + window_size, y : y + window_size, ...] = _worker(chunk)
            elif x == x_range[-1]:
                chunk = x_end[:, :, y : y + window_size, ...]
                arr[:, x : x + window_size, y : y + window_size, ...] = _worker(chunk)
            elif y != y_range[-1]:
                chunk = y_end[:, x : x + window_size, :, ...]
                arr[:, x : x + window_size, y : y + window_size, ...] = _worker(chunk)

    return arr
