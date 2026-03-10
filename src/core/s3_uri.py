"""S3 URI parsing helpers.

This module centralizes S3 URI parsing for ingest and store layers.
It keeps URI validation behavior consistent across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.errors import CrucibleIngestError, CrucibleStoreError


@dataclass(frozen=True)
class S3Location:
    """Parsed S3 location model."""

    bucket: str
    prefix: str


def parse_s3_uri(uri: str, domain: str) -> S3Location:
    """Parse and validate an S3 URI.

    Args:
        uri: URI in format ``s3://bucket/prefix``.
        domain: Error domain string ("ingest" or "store").

    Returns:
        Parsed bucket and prefix pair.

    Raises:
        CrucibleIngestError: For ingest-domain parse failures.
        CrucibleStoreError: For store-domain parse failures.
    """
    stripped_uri = uri.removeprefix("s3://")
    if "/" not in stripped_uri:
        _raise_uri_error(uri, domain)
    bucket, prefix = stripped_uri.split("/", 1)
    if not bucket or not prefix:
        _raise_uri_error(uri, domain)
    return S3Location(bucket=bucket, prefix=prefix)


def _raise_uri_error(uri: str, domain: str) -> None:
    """Raise a domain-specific invalid URI error.

    Args:
        uri: Invalid URI value.
        domain: Error domain string.

    Raises:
        CrucibleIngestError: For ingest domain.
        CrucibleStoreError: For store domain.
    """
    message = (
        f"Invalid S3 URI '{uri}': expected s3://bucket/prefix. "
        "Provide both bucket and prefix."
    )
    if domain == "ingest":
        raise CrucibleIngestError(message)
    raise CrucibleStoreError(message)
