"""Dataset manifest persistence helpers.

This module isolates JSON manifest IO for the flat dataset layout.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from core.constants import MANIFEST_FILE_NAME
from core.errors import ForgeStoreError
from core.types import DatasetManifest


def write_manifest_file(
    dataset_dir: Path,
    manifest: DatasetManifest,
    lance_written: bool,
) -> None:
    """Write dataset manifest file.

    Args:
        dataset_dir: Dataset directory.
        manifest: Manifest payload.
        lance_written: Whether Lance dataset was created.
    """
    manifest_dict = asdict(manifest)
    manifest_dict["created_at"] = manifest.created_at.isoformat()
    manifest_dict["lance_written"] = lance_written
    manifest_path = dataset_dir / MANIFEST_FILE_NAME
    manifest_path.write_text(json.dumps(manifest_dict, indent=2) + "\n", encoding="utf-8")


def read_manifest_file(manifest_path: Path) -> DatasetManifest:
    """Read and validate dataset manifest payload.

    Args:
        manifest_path: Manifest JSON path.

    Returns:
        Typed dataset manifest.

    Raises:
        ForgeStoreError: If manifest is missing or invalid.
    """
    if not manifest_path.exists():
        raise ForgeStoreError(
            f"Dataset manifest not found at {manifest_path}. "
            "Ingest data before requesting dataset info."
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ForgeStoreError(
            f"Failed to parse dataset manifest at {manifest_path}: {error.msg}."
        ) from error
    if not isinstance(payload, dict):
        raise ForgeStoreError(
            f"Failed to parse dataset manifest at {manifest_path}: "
            "expected JSON object at top level."
        )
    return manifest_from_dict(payload)


def manifest_from_dict(payload: dict[str, object]) -> DatasetManifest:
    """Deserialize manifest payload from dictionary.

    Args:
        payload: Manifest dictionary.

    Returns:
        Typed dataset manifest.
    """
    return DatasetManifest(
        dataset_name=str(payload["dataset_name"]),
        created_at=datetime.fromisoformat(str(payload["created_at"])),
        record_count=int(payload["record_count"]),
        source_uri=payload.get("source_uri"),
    )
