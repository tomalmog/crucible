"""S3 export helpers for snapshot payloads.

This module encapsulates boto3 client creation and directory upload.
It is shared by snapshot export operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import CrucibleConfig
from core.errors import CrucibleDependencyError, CrucibleStoreError


def create_s3_client(config: CrucibleConfig) -> Any:
    """Create boto3 S3 client for exports.

    Args:
        config: Runtime config with optional session settings.

    Returns:
        Boto3 S3 client.

    Raises:
        CrucibleDependencyError: If boto3 is missing.
    """
    try:
        import boto3
    except ImportError as error:
        raise CrucibleDependencyError(
            "S3 export requires boto3, but it is not installed. "
            "Install boto3 to export versions to s3:// destinations."
        ) from error
    session_kwargs: dict[str, str] = {}
    if config.s3_profile:
        session_kwargs["profile_name"] = config.s3_profile
    if config.s3_region:
        session_kwargs["region_name"] = config.s3_region
    session = boto3.session.Session(**session_kwargs)
    return session.client("s3")


def upload_directory(s3_client: Any, version_dir: Path, bucket: str, prefix: str) -> None:
    """Upload all version files to S3.

    Args:
        s3_client: Boto3 S3 client.
        version_dir: Local version directory.
        bucket: Destination bucket.
        prefix: Destination key prefix.

    Raises:
        CrucibleStoreError: If upload fails.
    """
    for local_file in sorted(version_dir.rglob("*")):
        if not local_file.is_file():
            continue
        relative_path = local_file.relative_to(version_dir)
        object_key = f"{prefix.rstrip('/')}/{relative_path.as_posix()}"
        try:
            s3_client.upload_file(str(local_file), bucket, object_key)
        except Exception as error:
            raise CrucibleStoreError(
                f"Failed to export snapshot file {local_file} to s3://{bucket}/{object_key}: {error}. "
                "Check AWS credentials and retry export."
            ) from error
