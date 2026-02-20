"""Shared obstore helpers extracted from loaders."""

from __future__ import annotations

import os
import tempfile
from typing import Union

import obstore as obs
from obstore.store import LocalStore, from_url


def from_dest(dest: str, *, region: str = "us-west-2"):
    """Build an obstore Store from an ``s3://`` URI or local path.

    This mirrors the pattern used across all four loaders.
    """
    if dest.startswith("s3://"):
        return from_url(dest, region=region)
    os.makedirs(dest, exist_ok=True)
    return LocalStore(prefix=dest)


def obstore_put_hkl(store, relpath: str, obj) -> None:
    """Serialize *obj* via hickle and write to *store* at *relpath*."""
    import hickle as hkl

    tmp = tempfile.NamedTemporaryFile(suffix=".hkl", delete=False)
    tmp.close()
    try:
        hkl.dump(obj, tmp.name, mode="w", compression="gzip")
        with open(tmp.name, "rb") as f:
            obs.put(store, relpath, f.read())
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass


def obstore_put_bytes(store, relpath: str, data: bytes) -> None:
    """Write raw bytes to *store* at *relpath*."""
    obs.put(store, relpath, data)
