"""Shared obstore helpers extracted from loaders."""

from __future__ import annotations

import os
import tempfile
from typing import Union

import obstore as obs
from obstore.store import LocalStore, from_url


def from_dest(dest: str, *, region: str = "us-east-1", profile: str | None = None):
    """Build an obstore Store from an ``s3://`` URI or local path.

    Uses Boto3CredentialProvider for S3 to inherit AWS credential chain
    (profiles, env vars, SSO, etc.).
    """
    if dest.startswith("s3://"):
        import boto3
        from obstore.auth.boto3 import Boto3CredentialProvider
        from obstore.store import S3Store

        session = boto3.Session(profile_name=profile)
        credential_provider = Boto3CredentialProvider(session)
        # Extract bucket name from s3://bucket/prefix
        bucket = dest.replace("s3://", "").split("/")[0]
        return S3Store(bucket, region=region, credential_provider=credential_provider)
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
