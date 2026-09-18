"""DSen2 bilinear / CNN 20 m -> 10 m super-resolution.

Ported from reference ``download_and_predict_job.py`` lines 125-177.
Requires a TensorFlow super-resolution session to be passed in.
Falls back to bilinear-only if no session is available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    # TYPE_CHECKING-only import — never executed at runtime, so this
    # doesn't add TensorFlow to the module's real import cost (this
    # module, like the other Lambda-worker code in this pipeline, avoids
    # eagerly importing TF). It's only here so the forward-referenced
    # "tf.compat.v1.Session" / "tf.Tensor" annotations below actually
    # resolve to something instead of mypy reporting `tf` as undefined.
    import tensorflow as tf


def superresolve_tile(
    arr: np.ndarray,
    sess: "tf.compat.v1.Session" | None = None,
    sr_logits: "tf.Tensor" | None = None,
    sr_inp: "tf.Tensor" | None = None,
    sr_inp_bilinear: "tf.Tensor" | None = None,
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

    if sr_logits is None or sr_inp is None or sr_inp_bilinear is None:
        # `sess` alone doesn't make the call runnable — sess.run() needs
        # all three tensors. Failing fast here with a clear message beats
        # letting sess.run() raise a confusing TF error deep inside the
        # windowing loop (or, worse, silently misbehaving if a future TF
        # version tolerates a None fetch/feed-dict key).
        raise ValueError(
            "superresolve_tile: sess was provided but sr_logits/sr_inp/"
            "sr_inp_bilinear were not — all four are required together, "
            "or none of them (for bilinear-only fallback)."
        )

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
                # `_worker` mutates `chunk` in place, and this branch's
                # chunk is a live VIEW into `arr` (not a copy), so `arr`
                # is already updated as soon as `_worker` runs — the
                # assignment below is a redundant self-write. That's only
                # safe because the edge branches (x_edge / y_edge, below)
                # always read from the `x_end`/`y_end` snapshots taken
                # BEFORE this loop started, never from live `arr`, and —
                # since x_range/y_range end with the edge position and
                # Python iterates lists in order — the edge windows
                # always run last for their row/column, so their
                # pristine-sourced writes are what's left standing in any
                # overlap region. Don't reorder this loop without
                # re-checking that invariant.
                chunk = arr[:, x : x + window_size, y : y + window_size, ...]
                arr[:, x : x + window_size, y : y + window_size, ...] = _worker(chunk)
            elif x == x_range[-1]:
                chunk = x_end[:, :, y : y + window_size, ...]
                arr[:, x : x + window_size, y : y + window_size, ...] = _worker(chunk)
            elif y != y_range[-1]:
                chunk = y_end[:, x : x + window_size, :, ...]
                arr[:, x : x + window_size, y : y + window_size, ...] = _worker(chunk)

    return arr
