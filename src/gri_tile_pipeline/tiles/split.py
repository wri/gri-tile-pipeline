"""Split a tiles CSV into chunked files, preserving the header row."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path


# Historical chunk-size ladder: first four chunks are small so early Lambda
# fan-outs surface failures fast, then 1600 each to keep per-chunk cost bounded.
DEFAULT_CHUNK_LADDER = [100, 200, 400, 800]
STEADY_CHUNK_SIZE = 1600


def _next_size(chunk_index: int, ladder: list[int], steady: int) -> int:
    if chunk_index <= len(ladder):
        return ladder[chunk_index - 1]
    return steady


def split_csv(
    path: str,
    *,
    chunk_size: int | None = None,
    encoding: str = "utf-8",
) -> list[str]:
    """Split *path* into `<base>_chunk_<n>.csv` files, returning the list written.

    If *chunk_size* is None, uses the historical ladder (100, 200, 400, 800, then
    1600 each). Otherwise every chunk has exactly *chunk_size* rows.
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    base = src.with_suffix("")
    ext = src.suffix or ".csv"

    ladder = [chunk_size] if chunk_size is not None else DEFAULT_CHUNK_LADDER
    steady = chunk_size if chunk_size is not None else STEADY_CHUNK_SIZE

    written: list[str] = []
    with src.open("r", encoding=encoding, newline="") as infile:
        reader = csv.reader(infile)
        try:
            header = next(reader)
        except StopIteration:
            print("Input CSV is empty. No chunks created.", file=sys.stderr)
            return written

        chunk_index = 1
        while True:
            size = _next_size(chunk_index, ladder, steady)
            out_path = f"{base}_chunk_{chunk_index}{ext}"
            rows_written = 0
            with open(out_path, "w", encoding=encoding, newline="") as outfile:
                writer = csv.writer(outfile)
                writer.writerow(header)
                for row in reader:
                    writer.writerow(row)
                    rows_written += 1
                    if rows_written >= size:
                        break
            if rows_written == 0:
                os.remove(out_path)
                break
            written.append(out_path)
            chunk_index += 1

    return written
