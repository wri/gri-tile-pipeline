"""Whittaker smoother for temporal regularisation.

Ported from reference ``preprocessing/whittaker_smoother.py``.
Uses sparse LU factorisation for O(n) per-pixel smoothing.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
from scipy.sparse import eye as speye
from scipy.sparse.linalg import splu


class WhittakerSmoother:
    """Second-order Whittaker smoother with pre-factored coefficient matrix.

    Args:
        lmbd: Smoothing parameter (higher = smoother).
        size: Number of input time steps.
        nbands: Number of spectral bands.
        dimx: Spatial X dimension.
        dimy: Spatial Y dimension.
        outsize: Number of output (monthly) time steps.
        average: If True, average smoothed values into ``outsize`` bins.
    """

    def __init__(
        self,
        lmbd: float,
        size: int,
        nbands: int = 14,
        dimx: int = 128,
        dimy: int = 128,
        outsize: int = 12,
        average: bool = True,
    ):
        self.lmbd = lmbd
        self.size = size
        self.nbands = nbands
        self.dimx = dimx
        self.dimy = dimy
        self.outsize = outsize
        self.average = average

        # Build second-order difference matrix
        diagonals = np.zeros(2 * 2 + 1, dtype=np.float32)
        diagonals[2] = 1.0
        for _ in range(2):
            diagonals = diagonals[:-1] - diagonals[1:]
        offsets = np.arange(2 + 1)
        shape = (self.size - 2, self.size)
        E = speye(self.size, format="csc", dtype=np.float32)
        D = scipy.sparse.diags(diagonals, offsets, shape, dtype=np.float32)
        D = D.conj().T.dot(D) * self.lmbd
        coefmat = E + D
        self.splu_coef = splu(coefmat)

    def smooth(self, y: np.ndarray) -> np.ndarray:
        """Apply Whittaker smoothing to a 1-D (or column-vectorised) array."""
        return self.splu_coef.solve(np.array(y))

    def interpolate_array(self, x: np.ndarray) -> np.ndarray:
        """Smooth a ``(T, H, W, B)`` array along the time axis.

        Returns ``(outsize, H, W, B)`` if ``average=True``, else ``(T, H, W, B)``.
        """
        x = np.reshape(x, (self.size, self.dimx * self.dimy * self.nbands))
        x = self.smooth(x)
        x = np.reshape(x, (self.size, self.dimx, self.dimy, self.nbands))

        if self.average:
            x = np.reshape(
                x, (12, x.shape[0] // 12, x.shape[1], x.shape[2], x.shape[3])
            )
            x = np.mean(x, axis=1)
        return x
