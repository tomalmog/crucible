"""Source document readers for ingestion.

This module loads text records from local paths or S3 prefixes.
It normalizes inputs into typed source records for transforms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from core.config import ForgeConfig
from core.constants import SUPPORTED_TEXT_EXTENSIONS
from core.errors import ForgeDependencyError, ForgeIngestError
from core.s3_uri import S3Location, parse_s3_uri
from core.types import SourceTextRecord


def read_source_records(source_uri: str, config: ForgeConfig) -> list[SourceTextRecord]:
    """Load source records from local files or S3.

    Args:
        source_uri: Local path, local file, or ``s3://`` URI.
        config: Runtime configuration for S3 session defaults.

    Returns:
        Ordered list of source records.

    Raises:
        ForgeIngestError: If source cannot be read.
    """
    if source_uri.startswith("s3://"):
        return _read_s3_records(source_uri, config)
    return _read_local_records(Path(source_uri).expanduser())


def _read_local_records(source_path: Path) -> list[SourceTextRecord]:
    """Read records from local file system.

    Args:
        source_path: Input file or directory.

    Returns:
        Collected source records.

    Raises:
        ForgeIngestError: If path is missing or unreadable.
    """
    if not source_path.exists():
        raise ForgeIngestError(
            f"Failed to read source at {source_path}: path does not exist. "
            "Provide an existing file or directory."
        )
    if source_path.is_file():
        return _read_file_records(source_path)
    records: list[SourceTextRecord] = []
    for file_path in sorted(source_path.rglob("*")):
        if file_path.is_file() and _is_supported_file(file_path):
            records.extend(_read_file_records(file_path))
    if not records:
        raise ForgeIngestError(
            f"No readable text files found under {source_path}. "
            f"Supported extensions: {SUPPORTED_TEXT_EXTENSIONS}."
        )
    return records


def _read_file_records(file_path: Path) -> list[SourceTextRecord]:
    """Read text records from a single file.

    Args:
        file_path: Path to text or JSONL file.

    Returns:
        List of parsed source records.
    """
    if file_path.suffix.lower() == ".jsonl":
        return _read_jsonl_records(file_path)
    text = file_path.read_text(encoding="utf-8")
    return [SourceTextRecord(source_uri=str(file_path), text=text)]


def _read_jsonl_records(file_path: Path) -> list[SourceTextRecord]:
    """Read ``text`` fields from JSONL input.

    Args:
        file_path: Path to JSONL file.

    Returns:
        Source records extracted from JSON lines.

    Raises:
        ForgeIngestError: If JSONL has invalid lines or missing text.
    """
    records: list[SourceTextRecord] = []
    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = _parse_jsonl_line(file_path, line, line_number)
        records.append(
            SourceTextRecord(source_uri=f"{file_path}:{line_number}", text=payload["text"])
        )
    return records


def _parse_jsonl_line(file_path: Path, line: str, line_number: int) -> dict[str, str]:
    """Parse and validate a JSONL row.

    Args:
        file_path: Parent file path for context.
        line: Raw JSON text line.
        line_number: One-based line number.

    Returns:
        Parsed JSON object containing text field.

    Raises:
        ForgeIngestError: If line is invalid.
    """
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        raise ForgeIngestError(
            f"Failed to parse JSONL record at {file_path}:{line_number}: "
            f"{error.msg}. Fix the JSON syntax and retry ingest."
        ) from error
    text_value = payload.get("text")
    if isinstance(text_value, str):
        return {"text": text_value}
    # Support prompt/response format (SFT, LoRA, etc.)
    prompt = payload.get("prompt")
    response = payload.get("response")
    if isinstance(prompt, str) and isinstance(response, str):
        return {"text": f"{prompt}\n{response}"}
    raise ForgeIngestError(
        f"Invalid JSONL record at {file_path}:{line_number}: "
        "expected 'text' field or 'prompt'/'response' fields. "
        "Fix the format and retry ingest."
    )


def _read_s3_records(source_uri: str, config: ForgeConfig) -> list[SourceTextRecord]:
    """Read records from S3 objects under a prefix.

    Args:
        source_uri: S3 prefix URI.
        config: Runtime configuration for region/profile.

    Returns:
        Source records loaded from objects.

    Raises:
        ForgeIngestError: If S3 read fails or no records found.
    """
    location = parse_s3_uri(source_uri, domain="ingest")
    s3_client = _create_s3_client(config)
    object_keys = _list_s3_keys(s3_client, location)
    records = _download_s3_text_records(s3_client, location.bucket, object_keys)
    if not records:
        raise ForgeIngestError(
            f"No readable text objects found for {source_uri}. "
            "Upload .txt/.md/.text/.jsonl files and retry ingest."
        )
    return records


def _create_s3_client(config: ForgeConfig) -> Any:
    """Create a boto3 S3 client.

    Args:
        config: Runtime config containing optional profile/region.

    Returns:
        Boto3 S3 client.

    Raises:
        ForgeDependencyError: If boto3 is missing.
    """
    try:
        import boto3
    except ImportError as error:
        raise ForgeDependencyError(
            "S3 support requires boto3, but it is not installed. "
            "Install boto3 to ingest from s3:// sources."
        ) from error
    session_kwargs = _build_boto3_session_kwargs(config)
    session = boto3.session.Session(**session_kwargs)
    return session.client("s3")


def _build_boto3_session_kwargs(config: ForgeConfig) -> dict[str, str]:
    """Build boto3 Session kwargs from config.

    Args:
        config: Runtime config.

    Returns:
        Session keyword arguments.
    """
    kwargs: dict[str, str] = {}
    if config.s3_profile:
        kwargs["profile_name"] = config.s3_profile
    if config.s3_region:
        kwargs["region_name"] = config.s3_region
    return kwargs


def _list_s3_keys(s3_client: Any, location: S3Location) -> list[str]:
    """List object keys under an S3 prefix.

    Args:
        s3_client: Boto3 S3 client.
        location: Target bucket/prefix.

    Returns:
        Sorted keys matching supported extensions.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=location.bucket, Prefix=location.prefix)
    keys: list[str] = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if _is_supported_key(key):
                keys.append(key)
    return sorted(keys)


def _download_s3_text_records(
    s3_client: Any,
    bucket: str,
    object_keys: Iterable[str],
) -> list[SourceTextRecord]:
    """Download text records from object keys.

    Args:
        s3_client: Boto3 S3 client.
        bucket: S3 bucket name.
        object_keys: Object keys to fetch.

    Returns:
        Loaded source records.
    """
    records: list[SourceTextRecord] = []
    for key in object_keys:
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        source_uri = f"s3://{bucket}/{key}"
        records.extend(_records_from_text_body(source_uri, key, body))
    return records


def _records_from_text_body(source_uri: str, key: str, body: str) -> list[SourceTextRecord]:
    """Parse text body into records based on object extension.

    Args:
        source_uri: Fully-qualified source URI.
        key: Object key for extension detection.
        body: Downloaded object body.

    Returns:
        Parsed records.

    Raises:
        ForgeIngestError: If JSONL content is invalid.
    """
    suffix = Path(key).suffix.lower()
    if suffix != ".jsonl":
        return [SourceTextRecord(source_uri=source_uri, text=body)]
    records: list[SourceTextRecord] = []
    for line_number, line in enumerate(body.splitlines(), 1):
        if not line.strip():
            continue
        payload = _parse_jsonl_line(Path(source_uri), line, line_number)
        records.append(
            SourceTextRecord(source_uri=f"{source_uri}:{line_number}", text=payload["text"])
        )
    return records


def _is_supported_file(file_path: Path) -> bool:
    """Return whether a local file extension is supported."""
    return file_path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS


def _is_supported_key(key: str) -> bool:
    """Return whether an S3 object key extension is supported."""
    return Path(key).suffix.lower() in SUPPORTED_TEXT_EXTENSIONS
