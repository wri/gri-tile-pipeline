"""Tests for the interpolated TTC color ramp (matches the legacy 2020
country-mosaic product's embedded color table at every 10th percent)."""

from gri_tile_pipeline.zonal.ttc_colormap import NODATA_VALUE, build_ttc_colormap


REFERENCE_STOPS = {
    0: (240, 247, 240, 255),
    10: (240, 247, 240, 255),
    20: (225, 241, 224, 255),
    30: (209, 235, 207, 255),
    40: (177, 219, 176, 255),
    50: (144, 202, 145, 255),
    60: (116, 182, 122, 255),
    70: (87, 161, 98, 255),
    80: (35, 127, 64, 255),
    90: (18, 116, 52, 255),
    100: (18, 116, 52, 255),
}


def test_has_all_256_entries():
    cmap = build_ttc_colormap()
    assert len(cmap) == 256
    assert set(cmap) == set(range(256))


def test_matches_reference_stops_exactly():
    cmap = build_ttc_colormap()
    for value, rgba in REFERENCE_STOPS.items():
        assert cmap[value] == rgba


def test_interpolates_between_stops():
    cmap = build_ttc_colormap()
    # Halfway between 20 (225,241,224) and 30 (209,235,207) should be
    # roughly the midpoint, not equal to either endpoint.
    mid = cmap[25]
    assert mid != REFERENCE_STOPS[20]
    assert mid != REFERENCE_STOPS[30]
    for lo, hi, m in zip(REFERENCE_STOPS[20], REFERENCE_STOPS[30], mid):
        assert min(lo, hi) <= m <= max(lo, hi)


def test_values_above_100_reuse_top_color():
    cmap = build_ttc_colormap()
    top = REFERENCE_STOPS[100]
    for v in (101, 150, 200, 254):
        assert cmap[v] == top


def test_nodata_is_fully_transparent():
    cmap = build_ttc_colormap()
    assert cmap[NODATA_VALUE] == (0, 0, 0, 0)
