"""Tests for data access functions."""
import pytest
import pystac
from gri_tile_pipeline.data_access import download_and_stack_assets
import xarray as xr

@pytest.fixture
def sample_stac_item():
    """Provides a single, valid STAC item dictionary for testing."""
    # This is a real Sentinel-2 L2A item from the planetary computer catalog.
    # We use a static dictionary to avoid network calls in tests.
    return {
        "stac_version": "1.0.0",
        "stac_extensions": [],
        "type": "Feature",
        "id": "S2B_MSIL2A_20230704T083559_N0509_R064_T36QYE_20230704T110631",
        "bbox": [33.339, -3.852, 34.42, -2.93],
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[33.339, -3.852], [34.42, -3.852], [34.42, -2.93], [33.339, -2.93], [33.339, -3.852]]],
        },
        "properties": {
            "datetime": "2023-07-04T08:35:59.024Z",
        },
        "collection": "sentinel-2-l2a",
        "links": [],
        "assets": {
            "B02": {
                "href": "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2a-cogs/36/Q/YE/2023/7/S2B_MSIL2A_20230704T083559_N0509_R064_T36QYE_20230704T110631.json"
            },
            "B03": {
                "href": "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2a-cogs/36/Q/YE/2023/7/S2B_MSIL2A_20230704T083559_N0509_R064_T36QYE_20230704T110631.json"
            },
            "B04": {
                "href": "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2a-cogs/36/Q/YE/2023/7/S2B_MSIL2A_20230704T083559_N0509_R064_T36QYE_20230704T110631.json"
            },
            "B08": {
                "href": "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2a-cogs/36/Q/YE/2023/7/S2B_MSIL2A_20230704T083559_N0509_R064_T36QYE_20230704T110631.json"
            }
        }
    }


def test_download_and_stack_assets(sample_stac_item):
    """Test that stackstac produces a correctly structured xarray object."""
    item_object = pystac.Item.from_dict(sample_stac_item)
    stack = download_and_stack_assets([item_object])

    assert stack is not None
    assert isinstance(stack, xr.DataArray)
    # Check for expected dimensions
    assert "time" in stack.dims
    assert "band" in stack.dims
    assert "y" in stack.dims
    assert "x" in stack.dims
    # Check that our requested bands are present
    assert all(band in stack.band.values for band in ["B02", "B03", "B04", "B08"])
    # Check that time dimension has one entry
    assert len(stack.time) == 1
