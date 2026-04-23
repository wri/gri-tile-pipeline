"""STAC Item builders for predict-tile outputs.

Pure functions — no I/O, no rasterio. Call from Lambda workers or local runs
to build a STAC 1.0.0 Item dict next to a predict GeoTIFF.
"""

from __future__ import annotations

from typing import Any


TILE_DEG = 1.0 / 18.0  # Tile grid spacing in degrees (EPSG:4326).

STAC_VERSION = "1.0.0"
STAC_COLLECTION = "gri-tree-cover-predictions"
STAC_PROJECTION_EXTENSION = (
    "https://stac-extensions.github.io/projection/v1.1.0/schema.json"
)
COG_MEDIA_TYPE = "image/tiff; application=geotiff; profile=cloud-optimized"


def tile_bbox(
    lon: float, lat: float, tile_deg: float = TILE_DEG
) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) for a tile centered on (lon, lat)."""
    half = tile_deg / 2.0
    return (lon - half, lat - half, lon + half, lat + half)


def bbox_to_geometry(bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    """Return a GeoJSON Polygon (closed CCW ring, lon-lat) for *bbox*."""
    west, south, east, north = bbox
    return {
        "type": "Polygon",
        "coordinates": [[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south],
        ]],
    }


def build_predict_stac_item(
    *,
    x_tile: int,
    y_tile: int,
    year: int,
    lon: float,
    lat: float,
    asset_href: str,
    model: dict[str, Any],
    pipeline_version: str,
    git_sha: str | None,
    run_id: str | None,
    created: str,
) -> dict[str, Any]:
    """Build a STAC 1.0.0 Item dict for a predict tile.

    Parameters
    ----------
    asset_href:
        Relative href of the COG (e.g. ``"1000X871Y_FINAL.tif"``). Kept
        relative so the item+asset pair is portable across buckets/prefixes.
    model:
        Dict with keys ``name``, ``path``, ``sha256`` (or None),
        ``input_size``, ``output_size``, ``length``.
    created:
        ISO-8601 UTC timestamp (``YYYY-MM-DDTHH:MM:SSZ``) for when the tile
        was produced. Callers should pass ``datetime.now(timezone.utc)``
        formatted.
    """
    tag = f"{x_tile}X{y_tile}Y"
    bbox = tile_bbox(lon, lat)

    return {
        "type": "Feature",
        "stac_version": STAC_VERSION,
        "stac_extensions": [STAC_PROJECTION_EXTENSION],
        "id": f"{tag}_{year}",
        "collection": STAC_COLLECTION,
        "geometry": bbox_to_geometry(bbox),
        "bbox": list(bbox),
        "properties": {
            "datetime": None,
            "start_datetime": f"{year}-01-01T00:00:00Z",
            "end_datetime": f"{year}-12-31T23:59:59Z",
            "created": created,
            "proj:epsg": 4326,
            "gri:tile_id": tag,
            "gri:x_tile": int(x_tile),
            "gri:y_tile": int(y_tile),
            "gri:year": int(year),
            "gri:pipeline_version": pipeline_version,
            "gri:git_sha": git_sha,
            "gri:run_id": run_id,
            "gri:model_name": model.get("name"),
            "gri:model_path": model.get("path"),
            "gri:model_sha256": model.get("sha256"),
            "gri:model_input_size": model.get("input_size"),
            "gri:model_output_size": model.get("output_size"),
            "gri:model_length": model.get("length"),
        },
        "assets": {
            "data": {
                "href": asset_href,
                "type": COG_MEDIA_TYPE,
                "roles": ["data"],
                "title": "Tree-cover prediction (uint8, %)",
            },
        },
        "links": [],
    }
