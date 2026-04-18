#!/usr/bin/env python3
"""
Split a CSV into chunk files, retaining the header in each chunk.

Chunk sizes:
  1: 100
  2: 200
  3: 400
  4: 800
  5+: 1600

Output naming:
  <original_filename>_chunk_<n>.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List


def chunk_sizes_sequence() -> List[int]:
    """Return the chunk sizes in order (first 4 special, then repeating 1600)."""
    return [100, 200, 400, 800]


def split_csv(path: str, encoding: str = "utf-8", newline: str = "") -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    base, ext = os.path.splitext(path)
    if not ext:
        ext = ".csv"

    sizes = chunk_sizes_sequence()
    chunk_index = 1

    with open(path, "r", encoding=encoding, newline=newline) as infile:
        reader = csv.reader(infile)

        try:
            header = next(reader)
        except StopIteration:
            # Empty file; nothing to write
            print("Input CSV is empty. No chunks created.", file=sys.stderr)
            return

        def next_chunk_size(i: int) -> int:
            # 1-based chunk index
            if i <= len(sizes):
                return sizes[i - 1]
            return 1600

        while True:
            size = next_chunk_size(chunk_index)
            rows_written = 0

            out_path = f"{base}_chunk_{chunk_index}{ext}"
            with open(out_path, "w", encoding=encoding, newline=newline) as outfile:
                writer = csv.writer(outfile)
                writer.writerow(header)

                for row in reader:
                    writer.writerow(row)
                    rows_written += 1
                    if rows_written >= size:
                        break

            if rows_written == 0:
                # We created a file with only the header but had no data left; remove it.
                os.remove(out_path)
                break

            print(f"Wrote {out_path} ({rows_written} data rows + header)")
            chunk_index += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a CSV into header-preserving chunk files.")
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding (default: utf-8). Try 'utf-8-sig' for Excel-exported CSVs.",
    )
    args = parser.parse_args()

    split_csv(args.csv_file, encoding=args.encoding)


if __name__ == "__main__":
    main()
