#!/usr/bin/env python3
# dem_download.py
# Args: --year --lon --lat --X_tile --Y_tile --dest [--expansion 300]
# Produces:
#   raw/misc/dem_{X}X{Y}Y.hkl   {"data": (H,W) float32}
# e.g. python scripts/dem_download.py --year 2022 --lon 1.3611 --lat -54.5278 --X_tile 999 --Y_tile 988 --dest out/simple_test/

from __future__ import annotations
import os
import sys
import argparse
import tempfile
import numpy as np
from loguru import logger
from math import sqrt
from pyproj import Transformer
from typing import Tuple

import obstore as obs
from obstore.store import from_url, LocalStore
import boto3
from scipy.ndimage import zoom

from pystac_client import Client
from odc.stac import stac_load, configure_rio
import hickle as hkl

from loaders.shared import make_bbox, obstore_put_hkl

EARTH_SEARCH_V1 = "https://earth-search.aws.element84.com/v1"
DEM_COLLECTION = "cop-dem-glo-30"  # Earth Search v1 Copernicus DEM 30m
DEM_REGION = "eu-central-1"

# configure_s3_access(profile=None, requester_pays=True, cloud_defaults=True)


def setup_logging(is_debug: bool) -> None:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if is_debug else "INFO")


def get_aws_principal() -> str:
    return boto3.client("sts").get_caller_identity()["Arn"]


def ensure_local_dirs_for_key(store, relpath: str) -> None:
    # Only create local directories for LocalStore to avoid remote '.keep' artifacts
    if isinstance(store, LocalStore):
        dirname = os.path.dirname(relpath)
        if dirname:
            os.makedirs(os.path.join(store.prefix, dirname), exist_ok=True)


def bbox2geojson(bbox: list) -> dict:
    coords = [
        [bbox[0], bbox[1]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]],
        [bbox[0], bbox[3]],
        [bbox[0], bbox[1]],
    ]
    return {"type": "Polygon", "coordinates": [coords]}

def bbox_4326_to_3857(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    tf = Transformer.from_crs(4326, 3857, always_xy=True)
    x0, y0 = tf.transform(bbox[0], bbox[1])
    x1, y1 = tf.transform(bbox[2], bbox[3])
    return (x0, y0, x1, y1)

def slopePython(inBlock, outBlock, inXSize, inYSize, zScale=1):
    """ Calculate slope using Python.
        If Numba is available will make use of autojit function
        to run at ~ 1/2 the speed of the Fortran module.
        If not will fall back to pure Python - which will be slow!
    """
    for x in range(1, inBlock.shape[2] - 1):
        for y in range(1, inBlock.shape[1] - 1):
            # Get window size
            dx = 2 * inXSize[y, x]
            dy = 2 * inYSize[y, x]

            # Calculate difference in elevation
            dzx = (inBlock[0, y, x - 1] - inBlock[0, y, x + 1]) * zScale
            dzy = (inBlock[0, y - 1, x] - inBlock[0, y + 1, x]) * zScale

            # Find normal vector to the plane
            nx = -1 * dy * dzx
            ny = -1 * dx * dzy
            nz = dx * dy

            slopeRad = np.arccos(nz / sqrt(nx**2 + ny**2 + nz**2))
            slopeDeg = (180. / np.pi) * slopeRad

            outBlock[0, y, x] = slopeDeg

    return outBlock


def slopePythonPlane(inBlock,
                     outBlock,
                     inXSize,
                     inYSize,
                     A_mat,
                     z_vec,
                     winSize=3,
                     zScale=1):
    """ Calculate slope using Python.
        Algorithm fits plane to a window of data and calculated the slope
        from this - slope than the standard algorithm but can deal with
        noisy data batter.
        The matrix A_mat (winSize**2,3) and vector zScale (winSize**2) are allocated
        outside the function and passed in.
    """

    winOffset = int(winSize / 2)

    for x in range(winOffset - 1, inBlock.shape[2]):
        for y in range(winOffset - 1, inBlock.shape[1]):
            # Get window size
            dx = winSize * inXSize[y, x]
            dy = winSize * inYSize[y, x]

            # Calculate difference in elevation
            """
                Solve A b = x to give x
                Where A is a matrix of:
                    x_pos | y_pos | 1
                and b is elevation
                and x are the coefficents
            """

            # Form matrix
            index = 0
            for i in range(-1 * winOffset, winOffset + 1):
                for j in range(-1 * winOffset, winOffset + 1):

                    A_mat[index, 0] = 0 + (i * inXSize[y, x])
                    A_mat[index, 1] = 0 + (j * inYSize[y, x])
                    A_mat[index, 2] = 1

                    # Elevation
                    z_vec[index] = inBlock[0, y + j, x + i] * zScale

                    index += 1

            # Linear fit
            coeff_vec = np.linalg.lstsq(A_mat, z_vec)[0]

            # Calculate dzx and dzy
            dzx = coeff_vec[0] * dx
            dzy = coeff_vec[1] * dy

            # Find normal vector to the plane
            nx = -1 * dy * dzx
            ny = -1 * dx * dzy
            nz = dx * dy

            slopeRad = np.arccos(nz / sqrt(nx**2 + ny**2 + nz**2))
            slopeDeg = (180. / np.pi) * slopeRad

            outBlock[0, y, x] = slopeDeg

    return outBlock


def calcSlope(inBlock,
              inXSize,
              inYSize,
              fitPlane=False,
              zScale=1,
              winSize=3,
              minSlope=None):
    """ Calculates slope for a block of data
        Arrays are provided giving the size for each pixel.
        * inBlock - In elevation
        * inXSize - Array of pixel sizes (x)
        * inYSize - Array of pixel sizes (y)
        * fitPlane - Calculate slope by fitting a plane to elevation
                     data using least squares fitting.
        * zScale - Scaling factor between horizontal and vertical
        * winSize - Window size to fit plane over.
    """
    # If fortran class could be imported use this
    # Otherwise run through loop in python (which will be slower)
    # Setup output block
    outBlock = np.zeros_like(inBlock, dtype=np.float32)
    if fitPlane:
        # Setup matrix and vector required for least squares fitting.
        winOffset = int(winSize / 2)
        A_mat = np.zeros((winSize**2, 3))
        z_vec = np.zeros(winSize**2)

        slopePythonPlane(inBlock, outBlock, inXSize, inYSize, A_mat, z_vec,
                         zScale, winSize)
    else:
        slopePython(inBlock, outBlock, inXSize, inYSize, zScale)

    if minSlope is not None:
        # Set very low values to constant
        outBlock[0] = np.where(
            np.logical_and(outBlock[0] > 0, outBlock[0] < minSlope), minSlope,
            outBlock[0])
    return outBlock

def main():
    ap = argparse.ArgumentParser(
        description="Copernicus DEM GLO-30 loader, legacy-compatible output."
    )
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--X_tile", type=int, required=True)
    ap.add_argument("--Y_tile", type=int, required=True)
    ap.add_argument("--dest", type=str, required=True)
    ap.add_argument("--expansion", type=int, default=300)
    ap.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = ap.parse_args()

    setup_logging(args.debug)

    try:
        _run_core(
            year=args.year, lon=args.lon, lat=args.lat,
            X_tile=args.X_tile, Y_tile=args.Y_tile,
            dest=args.dest, expansion=args.expansion,
            debug=args.debug,
        )
    except SystemExit:
        raise
    except Exception:
        logger.exception("DEM slope generation failed")
        raise SystemExit(1)


# --- programmatic entrypoint for Lithops / local ---
def _run_core(
    year: int,
    lon: float,
    lat: float,
    X_tile: int,
    Y_tile: int,
    dest: str,
    expansion: int = 300,
    debug: bool = False,
) -> None:
    """Core DEM download logic, called by both main() and run()."""
    if debug:
        setup_logging(True)

    logger.info(
        f"Starting DEM job year={year} tile={X_tile}X{Y_tile}Y dest={dest}"
    )
    if dest.startswith("s3://"):
        store = from_url(dest, region="us-east-1")
    else:
        os.makedirs(dest, exist_ok=True)
        store = LocalStore(prefix=dest)

    configure_rio(
        cloud_defaults=True, aws={"requester_pays": True, "region_name": DEM_REGION}
    )

    try:
        logger.debug(f"AWS Principal: {get_aws_principal()}")
    except Exception:
        logger.warning("Could not retrieve AWS principal")

    initial_bbx = [lon, lat, lon, lat]
    bbx = make_bbox(initial_bbx, expansion=expansion / 30)
    logger.debug(f"BBX: {bbx}")
    geo_bbx = bbox2geojson(bbx)

    base_key = f"{year}/raw/{X_tile}/{Y_tile}/raw"
    misc_key = f"{base_key}/misc"
    fn_dem_key = f"{misc_key}/dem_{X_tile}X{Y_tile}Y.hkl"
    ensure_local_dirs_for_key(store, fn_dem_key)

    client = Client.open(EARTH_SEARCH_V1)
    search = client.search(collections=[DEM_COLLECTION], bbox=bbx)
    logger.debug(f"Search: {search.url_with_parameters()}")
    items = search.item_collection()
    logger.info(f"Found {len(items)} DEM items")
    if len(items) == 0:
        raise RuntimeError(f"No DEM items found for bbox={bbx}")

    ds = stac_load(
        items,
        bands=["data"],
        geopolygon=geo_bbx,
        resampling="bilinear",
        chunks={},
    )
    logger.debug(f"DEM Dataset: {ds}")
    for dim_name, size in ds.sizes.items():
        logger.debug(f"- {dim_name}: {size}")
    dem = (
        ds["data"]
        .isel(time=0)
        .transpose("latitude", "longitude")
        .values.astype("float32")
    )
    if not np.isfinite(dem).all():
        raise ValueError("DEM contains non-finite values after load")
    logger.debug(f"DEM stats: min={dem.min()}, max={dem.max()}, mean={dem.mean()}, std={dem.std()}")

    dem = zoom(dem, 3, order=1)

    width = dem.shape[0]
    height = dem.shape[1]
    logger.debug(f"DEM shape: {dem.shape}")

    dem = calcSlope(dem.reshape((1, width, height)),
                          np.full((width, height), 10),
                          np.full((width, height), 10),
                          zScale=1,
                          minSlope=0.02)
    dem = dem.reshape((width, height, 1))

    dem = dem[1:width - 1, 1:height - 1, :]
    dem = dem.squeeze()
    if not np.isfinite(dem).all():
        raise ValueError("Slope DEM contains non-finite values")
    logger.debug(f"Slope stats: min={dem.min()}, max={dem.max()}, mean={dem.mean()}, std={dem.std()}")

    obstore_put_hkl(store, fn_dem_key, dem)
    full_path = f"{dest.rstrip('/')}/{fn_dem_key}"
    logger.info(
        f"Completed DEM slope generation shape={dem.shape} saved={full_path}"
    )


def run(
    year: int | str,
    lon: float,
    lat: float,
    X_tile: int | str,
    Y_tile: int | str,
    dest: str,
    expansion: int = 300,
    debug: bool = False,
) -> dict:
    """Programmatic entry-point for Lithops and local execution."""
    _run_core(
        year=int(year), lon=float(lon), lat=float(lat),
        X_tile=int(X_tile), Y_tile=int(Y_tile),
        dest=dest, expansion=expansion, debug=debug,
    )
    return {
        "product": "dem",
        "year": int(year),
        "lon": float(lon),
        "lat": float(lat),
        "X_tile": int(X_tile),
        "Y_tile": int(Y_tile),
        "dest": dest,
    }

if __name__ == "__main__":
    main()
