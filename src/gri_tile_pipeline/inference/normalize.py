"""Per-band min/max normalization constants and normalization function.

Constants from reference ``download_and_predict_job.py`` lines 1950-1963.

Band order (17 channels):
  0-9: S2 bands (Blue, Green, Red, NIR, RE1, RE2, RE3, NIR08, SWIR16, SWIR22)
  10:  DEM (elevation)
  11:  S1 VV
  12:  S1 VH
  13:  EVI
  14:  BI
  15:  MSAVI2
  16:  GRNDVI
"""

from __future__ import annotations

import numpy as np

# fmt: off
MIN_ALL = np.array([
    0.006576638437476157, 0.0162050812542916, 0.010040436408026246,
    0.013351644159609368, 0.01965362020294499, 0.014229037918669413,
    0.015289539940489814, 0.011993591210803388, 0.008239871824216068,
    0.006546120393682765, 0.0, 0.0, 0.0, -0.1409399364817101,
    -0.4973397113668104, -0.09731556326714398, -0.7193834232943873,
], dtype=np.float32)

MAX_ALL = np.array([
    0.2691233691920348, 0.3740291447318227, 0.5171435111009385,
    0.6027466239414053, 0.5650263218127718, 0.5747005416952773,
    0.5933928435187305, 0.6034943160143434, 0.7472037842374304,
    0.7000076295109483,
    0.4,
    0.948334642387533,
    0.6729257769285485, 0.8177635298774327, 0.35768999002433816,
    0.7545951919107605, 0.7602693339366691,
], dtype=np.float32)
# fmt: on


def normalize_subtile(subtile: np.ndarray) -> np.ndarray:
    """Normalize a subtile to [-1, 1] using global min/max constants.

    Args:
        subtile: ``(..., 17)`` array — last axis is the 17-channel feature stack.

    Returns:
        Normalized array (same shape, float32).
    """
    subtile = subtile.astype(np.float32, copy=True)
    for band in range(subtile.shape[-1]):
        mins = MIN_ALL[band]
        maxs = MAX_ALL[band]
        subtile[..., band] = np.clip(subtile[..., band], mins, maxs)
        midrange = (maxs + mins) / 2
        rng = maxs - mins
        subtile[..., band] = (subtile[..., band] - midrange) / (rng / 2)
    return subtile
