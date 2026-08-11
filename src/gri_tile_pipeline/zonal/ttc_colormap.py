"""TTC color ramp matching the legacy 2020 country-mosaic product.

Extracted from ``s3://tof-output/2020/mosaics/Ethiopia.tif``'s embedded
color table: a 9-class sequential green ramp (ColorBrewer-style Greens)
defined only at every 10th percent. That table leaves every other value
black -- fine for the legacy pipeline, whose values were pre-binned to the
nearest 10, but our predictions are continuous 0-100 floats. The same 9
color stops are linearly interpolated across every integer percent here
instead of repeated verbatim.
"""

from __future__ import annotations

NODATA_VALUE = 255

RGBA = tuple[int, int, int, int]

# (percent, RGBA) control points, read directly off the legacy product's
# embedded GeoTIFF color table via rasterio's ``colormap(1)``.
_STOPS: list[tuple[int, RGBA]] = [
    (0, (240, 247, 240, 255)),
    (10, (240, 247, 240, 255)),
    (20, (225, 241, 224, 255)),
    (30, (209, 235, 207, 255)),
    (40, (177, 219, 176, 255)),
    (50, (144, 202, 145, 255)),
    (60, (116, 182, 122, 255)),
    (70, (87, 161, 98, 255)),
    (80, (35, 127, 64, 255)),
    (90, (18, 116, 52, 255)),
    (100, (18, 116, 52, 255)),
]


def _lerp_color(c0: RGBA, c1: RGBA, t: float) -> RGBA:
    return tuple(round(a + (b - a) * t) for a, b in zip(c0, c1))  # type: ignore[return-value]


def build_ttc_colormap() -> dict[int, RGBA]:
    """Build a full 0-255 colormap by interpolating the legacy ramp.

    Values 0-100 are linearly interpolated between the 9-color reference
    stops. Values 101-254 are unreachable for a valid TTC percentage but
    are filled with the 100% color for completeness. 255 (NoData) is
    fully transparent, matching the reference file.
    """
    cmap: dict[int, RGBA] = {}
    for (lo_v, lo_c), (hi_v, hi_c) in zip(_STOPS, _STOPS[1:]):
        span = hi_v - lo_v
        for v in range(lo_v, hi_v):
            t = (v - lo_v) / span if span else 0.0
            cmap[v] = _lerp_color(lo_c, hi_c, t)

    top_color = _STOPS[-1][1]
    for v in range(100, NODATA_VALUE):
        cmap[v] = top_color
    cmap[NODATA_VALUE] = (0, 0, 0, 0)
    return cmap
