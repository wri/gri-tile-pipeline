"""Shared obstore helpers extracted from loaders."""

from __future__ import annotations

import os
import tempfile
from typing import Union

import obstore as obs
from obstore.store import LocalStore, from_url


def make_s3_store(bucket: str, region: str, profile: str | None = None):
    """Build an obstore S3Store authenticated via boto3's credential chain.

    obstore doesn't read ``AWS_PROFILE`` natively, so we delegate credential
    resolution to boto3 (respects profiles, env vars, SSO, IAM role).
    """
    import boto3
    from obstore.auth.boto3 import Boto3CredentialProvider
    from obstore.store import S3Store

    session = boto3.Session(profile_name=profile)
    credential_provider = Boto3CredentialProvider(session)
    return S3Store(bucket, region=region, credential_provider=credential_provider)


def from_dest(dest: str, *, region: str = "us-east-1", profile: str | None = None):
    """Build an obstore Store from an ``s3://`` URI or local path.

    Uses boto3's credential chain for S3 (profiles, env vars, SSO, etc.) via
    ``Boto3CredentialProvider``, since obstore doesn't read ``AWS_PROFILE``
    natively.

    ``dest`` may include a path beyond the bucket (e.g.
    ``s3://bucket/sentinel/PROJECT``) — the whole URI is passed to
    ``S3Store.from_url`` so the returned store is scoped to that prefix and
    a subsequent ``obs.get(store, "2024/tiles/...")`` resolves under it.
    Previously this only kept the bucket name (``.split("/")[0]``) and
    silently discarded everything else, so callers using a project-scoped
    ``dest`` (e.g. ``sentinel_flow``'s ``sentinel/{project_short_name}``)
    would read/write at the bucket root instead — collapsing distinct
    projects onto the same global tile keyspace.
    """
    if dest.startswith("s3://"):
        import boto3
        from obstore.auth.boto3 import Boto3CredentialProvider
        from obstore.store import S3Store

        session = boto3.Session(profile_name=profile)
        credential_provider = Boto3CredentialProvider(session)
        return S3Store.from_url(dest, region=region, credential_provider=credential_provider)
    os.makedirs(dest, exist_ok=True)
    return LocalStore(prefix=dest)


def validate_aws(store, *, probe_key: str = "__aws_probe__") -> None:
    """Probe *store* to surface auth/permission errors before fanning out.

    Raises the underlying obstore exception if the store cannot satisfy a
    basic head/get on *probe_key* (which is allowed to be missing — we only
    care that the call doesn't fail with an auth error).
    """
    try:
        obs.head(store, probe_key)
    except Exception as exc:
        msg = str(exc).lower()
        # A missing key is fine; anything else (403, 401, invalid creds, etc.)
        # indicates a real credentials or configuration problem.
        if "not found" in msg or "nosuchkey" in msg or "404" in msg:
            return
        raise


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
