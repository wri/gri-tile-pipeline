#!/usr/bin/env python3
"""Generate subregions_conf.geojson by joining Natural Earth boundaries with region_conf.csv.

Usage:
    python scripts/generate_subregions_geojson.py

Output:
    src/gri_tile_pipeline/zonal/data/subregions_conf.geojson
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import os
import tempfile
import urllib.request
import zipfile


def main():
    data_dir = Path(__file__).parent.parent / "src" / "gri_tile_pipeline" / "zonal" / "data"
    region_conf = pd.read_csv(data_dir / "region_conf.csv")

    # Download Natural Earth 110m countries (has SUBREGION column)
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    tmp = tempfile.mkdtemp()
    zip_path = os.path.join(tmp, "ne.zip")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    shp = [f for f in os.listdir(tmp) if f.endswith(".shp")][0]
    world = gpd.read_file(os.path.join(tmp, shp))

    # Filter to only subregions present in region_conf.csv
    target_subregions = set(region_conf["category"])
    filtered = world[world["SUBREGION"].isin(target_subregions)].copy()

    # Dissolve country polygons into subregion polygons
    subregions = filtered.dissolve(by="SUBREGION").reset_index()

    # Join with region_conf statistics
    merged = subregions.merge(
        region_conf[["category", "p_lower_95", "p_upper_95", "r_lower_95", "r_upper_95"]],
        left_on="SUBREGION",
        right_on="category",
        how="inner",
    )

    # Select only needed columns
    result = gpd.GeoDataFrame(
        merged[["category", "p_lower_95", "p_upper_95", "r_lower_95", "r_upper_95", "geometry"]],
        geometry="geometry",
        crs=world.crs,
    )

    output_path = data_dir / "subregions_conf.geojson"
    result.to_file(output_path, driver="GeoJSON")
    print(f"Generated {output_path} with {len(result)} subregions:")
    for _, row in result.iterrows():
        print(f"  {row['category']}: p=[{row['p_lower_95']:.4f}, {row['p_upper_95']:.4f}] r=[{row['r_lower_95']:.4f}, {row['r_upper_95']:.4f}]")


if __name__ == "__main__":
    main()
