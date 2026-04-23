"""Tests for spatial clustering of polygons by shared tiles."""

import pandas as pd
import pytest
from shapely.geometry import box

from gri_tile_pipeline.zonal.spatial_clustering import UnionFind, cluster_polygons_by_tiles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gdf(polygons):
    """Build a GeoDataFrame from a list of Shapely geometries."""
    import geopandas as gpd
    return gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")


def _make_tile_lookup(tile_coords):
    """Build a tile lookup DataFrame from (x, y) center coordinates.

    Tile grid indices are computed as round(x * 18) and round(y * 18).
    """
    rows = []
    for x, y in tile_coords:
        rows.append({
            "X": x,
            "Y": y,
            "X_tile": round(x * 18),
            "Y_tile": round(y * 18),
        })
    return pd.DataFrame(rows)


# A tile is 1/18 degree wide.  Build a tile grid around the origin
# with 4 tiles: (0,0), (1/18,0), (0,1/18), (1/18,1/18)
TILE_SIZE = 1 / 18
HALF = TILE_SIZE / 2

# Tile centers
T00 = (0.0, 0.0)
T10 = (TILE_SIZE, 0.0)
T01 = (0.0, TILE_SIZE)
T11 = (TILE_SIZE, TILE_SIZE)

# Far-away tile
T_FAR = (10.0, 10.0)

TILE_LOOKUP = _make_tile_lookup([T00, T10, T01, T11, T_FAR])


# ---------------------------------------------------------------------------
# UnionFind tests
# ---------------------------------------------------------------------------

class TestUnionFind:
    def test_initial_state(self):
        uf = UnionFind(5)
        # Each element is its own component
        assert len(uf.components()) == 5

    def test_union_two(self):
        uf = UnionFind(3)
        uf.union(0, 1)
        comps = uf.components()
        assert len(comps) == 2
        # 0 and 1 are in the same component
        for indices in comps.values():
            if 0 in indices:
                assert 1 in indices

    def test_transitive(self):
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(1, 2)
        comps = uf.components()
        assert len(comps) == 1
        assert sorted(list(comps.values())[0]) == [0, 1, 2, 3]

    def test_idempotent(self):
        uf = UnionFind(3)
        uf.union(0, 1)
        uf.union(0, 1)
        assert len(uf.components()) == 2


# ---------------------------------------------------------------------------
# cluster_polygons_by_tiles tests
# ---------------------------------------------------------------------------

class TestClusterPolygonsByTiles:
    def test_single_polygon(self):
        """One polygon overlapping tile T00 → 1 cluster."""
        poly = box(-HALF, -HALF, HALF, HALF)  # centered on T00
        gdf = _make_gdf([poly])
        clusters = cluster_polygons_by_tiles(gdf, TILE_LOOKUP)

        assert len(clusters) == 1
        cluster_gdf, tiles_df = clusters[0]
        assert len(cluster_gdf) == 1
        assert len(tiles_df) >= 1

    def test_two_polygons_same_tile(self):
        """Two polygons overlapping the same tile → 1 cluster."""
        p1 = box(-HALF * 0.5, -HALF * 0.5, HALF * 0.5, HALF * 0.5)
        p2 = box(-HALF * 0.3, -HALF * 0.3, HALF * 0.3, HALF * 0.3)
        gdf = _make_gdf([p1, p2])
        clusters = cluster_polygons_by_tiles(gdf, TILE_LOOKUP)

        assert len(clusters) == 1
        assert len(clusters[0][0]) == 2

    def test_two_disjoint_polygons(self):
        """Two polygons on distant tiles → 2 clusters."""
        p_near = box(-HALF, -HALF, HALF, HALF)          # near T00
        p_far = box(10.0 - HALF, 10.0 - HALF, 10.0 + HALF, 10.0 + HALF)  # near T_FAR
        gdf = _make_gdf([p_near, p_far])
        clusters = cluster_polygons_by_tiles(gdf, TILE_LOOKUP)

        assert len(clusters) == 2
        sizes = sorted([len(c[0]) for c in clusters])
        assert sizes == [1, 1]

    def test_transitive_merge(self):
        """A overlaps T00, B overlaps T00+T10, C overlaps T10 → 1 cluster."""
        p_a = box(-HALF, -HALF, HALF * 0.5, HALF * 0.5)  # only T00
        p_b = box(-HALF * 0.2, -HALF * 0.2, TILE_SIZE + HALF * 0.2, HALF * 0.2)  # spans T00 and T10
        p_c = box(TILE_SIZE - HALF * 0.5, -HALF * 0.5, TILE_SIZE + HALF, HALF * 0.5)  # only T10
        gdf = _make_gdf([p_a, p_b, p_c])
        clusters = cluster_polygons_by_tiles(gdf, TILE_LOOKUP)

        assert len(clusters) == 1
        assert len(clusters[0][0]) == 3

    def test_empty_gdf(self):
        """Empty GeoDataFrame → empty result."""
        import geopandas as gpd
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        clusters = cluster_polygons_by_tiles(gdf, TILE_LOOKUP)
        assert clusters == []

    def test_no_intersecting_tiles(self):
        """Polygon in a region with no tiles → empty result."""
        # Far from any tile in the lookup
        poly = box(50.0, 50.0, 50.1, 50.1)
        gdf = _make_gdf([poly])
        clusters = cluster_polygons_by_tiles(gdf, TILE_LOOKUP)
        assert clusters == []

    def test_tiles_df_has_correct_columns(self):
        """Returned tiles_df has X_tile, Y_tile, X, Y columns."""
        poly = box(-HALF, -HALF, HALF, HALF)
        gdf = _make_gdf([poly])
        clusters = cluster_polygons_by_tiles(gdf, TILE_LOOKUP)
        _, tiles_df = clusters[0]
        assert set(tiles_df.columns) >= {"X_tile", "Y_tile", "X", "Y"}

    def test_sorted_largest_first(self):
        """Clusters are sorted largest first."""
        p1 = box(-HALF, -HALF, HALF, HALF)
        p2 = box(-HALF * 0.5, -HALF * 0.5, HALF * 0.5, HALF * 0.5)
        p3 = box(10.0 - HALF, 10.0 - HALF, 10.0 + HALF, 10.0 + HALF)
        gdf = _make_gdf([p1, p2, p3])
        clusters = cluster_polygons_by_tiles(gdf, TILE_LOOKUP)

        assert len(clusters) == 2
        # First cluster should have 2 polygons (p1, p2 share T00)
        assert len(clusters[0][0]) == 2
        assert len(clusters[1][0]) == 1
