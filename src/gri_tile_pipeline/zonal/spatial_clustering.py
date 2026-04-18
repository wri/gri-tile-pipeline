"""Spatial clustering of polygons by shared prediction tiles.

Groups polygons into compact spatial clusters using a union-find on
shared tile keys, then processes each cluster independently to avoid
building enormous mosaics for geographically scattered projects.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

import pandas as pd
from loguru import logger


class UnionFind:
    """Disjoint-set with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def components(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            groups[self.find(i)].append(i)
        return dict(groups)


def cluster_polygons_by_tiles(
    gdf: "gpd.GeoDataFrame",
    global_lookup: pd.DataFrame,
    tile_size: float = 1 / 18,
) -> List[Tuple["gpd.GeoDataFrame", pd.DataFrame]]:
    """Group polygons into spatial clusters based on shared tiles.

    Uses a single DuckDB spatial join (all polygons x all tiles) followed
    by a union-find on shared tile keys to identify connected components.

    Args:
        gdf: Polygon GeoDataFrame (must have a geometry column).
        global_lookup: Tile lookup with columns X, Y, X_tile, Y_tile.
        tile_size: Tile width in degrees (default 1/18).

    Returns:
        List of ``(polygon_subset_gdf, tiles_df)`` tuples, sorted
        largest cluster first.  ``tiles_df`` has columns
        ``X_tile, Y_tile, X, Y`` (deduplicated).
    """
    import duckdb
    from shapely import to_wkb

    if gdf.empty:
        return []

    n = len(gdf)
    half = tile_size / 2

    # Prepare polygon table with sequential indices and WKB geometry
    poly_df = pd.DataFrame({
        "poly_idx": range(n),
        "geom_wkb": gdf.geometry.reset_index(drop=True).apply(
            lambda g: bytes(to_wkb(g))
        ),
    })

    con = duckdb.connect()
    con.install_extension("spatial")
    con.load_extension("spatial")
    con.execute("SET geometry_always_xy=true")

    con.register("polygons", poly_df)
    con.register("tiles", global_lookup)

    # Single batch spatial join: all polygons x all tiles
    pairs = con.execute(f"""
        SELECT p.poly_idx, t.X_tile, t.Y_tile, t.X, t.Y
        FROM polygons p, tiles t
        WHERE ST_Intersects(
            ST_GeomFromWKB(p.geom_wkb),
            ST_MakeEnvelope(
                t.X - {half}, t.Y - {half},
                t.X + {half}, t.Y + {half}
            )
        )
    """).fetch_df()

    con.close()

    if pairs.empty:
        logger.warning("No polygon-tile intersections found")
        return []

    # Cast tile columns to int
    pairs["X_tile"] = pairs["X_tile"].astype(int)
    pairs["Y_tile"] = pairs["Y_tile"].astype(int)
    pairs["poly_idx"] = pairs["poly_idx"].astype(int)

    # Build tile → polygon index mapping
    tile_to_polys: dict[tuple[int, int], list[int]] = defaultdict(list)
    for poly_idx, x_tile, y_tile in zip(
        pairs["poly_idx"], pairs["X_tile"], pairs["Y_tile"]
    ):
        tile_to_polys[(x_tile, y_tile)].append(poly_idx)

    # Union polygons that share tiles
    uf = UnionFind(n)
    for poly_indices in tile_to_polys.values():
        for j in range(1, len(poly_indices)):
            uf.union(poly_indices[0], poly_indices[j])

    components = uf.components()

    # Build per-cluster (gdf subset, tiles_df)
    # Pre-index the pairs DataFrame for fast per-polygon lookup
    poly_idx_to_tiles = pairs.groupby("poly_idx")[["X_tile", "Y_tile", "X", "Y"]].apply(
        lambda g: g.values.tolist()
    )

    clusters: List[Tuple["gpd.GeoDataFrame", pd.DataFrame]] = []
    gdf_reset = gdf.reset_index(drop=True)

    for _root, indices in components.items():
        cluster_gdf = gdf_reset.iloc[indices].copy()

        # Gather deduplicated tiles for this cluster
        seen: set[tuple[int, int]] = set()
        tile_rows: list[dict] = []
        for idx in indices:
            if idx not in poly_idx_to_tiles.index:
                continue
            for xt, yt, x, y in poly_idx_to_tiles[idx]:
                key = (int(xt), int(yt))
                if key not in seen:
                    seen.add(key)
                    tile_rows.append({"X_tile": int(xt), "Y_tile": int(yt), "X": x, "Y": y})

        tiles_df = pd.DataFrame(tile_rows) if tile_rows else pd.DataFrame(
            columns=["X_tile", "Y_tile", "X", "Y"]
        )
        clusters.append((cluster_gdf, tiles_df))

    # Sort largest cluster first
    clusters.sort(key=lambda c: len(c[0]), reverse=True)

    logger.debug(
        f"Clustering: {n} polygons → {len(clusters)} cluster(s), "
        f"sizes={[len(c[0]) for c in clusters]}"
    )
    return clusters
