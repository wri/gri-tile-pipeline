import os

from lithops_job_tracker import _read_tiles_csv


def test_read_tiles_csv():
    tile_fiole_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_tiles.csv')
    tiles = _read_tiles_csv(tile_fiole_path)
    assert len(tiles) == 3
    assert tiles[0]['year'] == 2025
