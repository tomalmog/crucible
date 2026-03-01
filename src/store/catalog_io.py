"""Catalog and manifest persistence helpers.

This module isolates JSON catalog IO and version id generation.
It keeps snapshot store orchestration focused on business flow.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from core.constants import MANIFEST_FILE_NAME
from core.errors import ForgeStoreError
from core.types import DataRecord, SnapshotManifest


def build_version_id(dataset_name: str, records: tuple[DataRecord, ...]) -> str:
    """Build deterministic version id from dataset and records.

    Args:
        dataset_name: Dataset identifier.
        records: Snapshot records.

    Returns:
        Version id string.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    digest_seed = "|".join(record.record_id for record in records)
    digest = hashlib.sha256(digest_seed.encode("utf-8")).hexdigest()[:10]
    return f"{dataset_name}-{timestamp}-{digest}"


def write_manifest_file(
    version_dir: Path,
    manifest: SnapshotManifest,
    lance_written: bool,
) -> None:
    """Write per-version manifest file.

    Args:
        version_dir: Snapshot version directory.
        manifest: Manifest payload.
        lance_written: Whether Lance dataset was created.
    """
    manifest_dict = asdict(manifest)
    manifest_dict["created_at"] = manifest.created_at.isoformat()
    manifest_dict["lance_written"] = lance_written
    manifest_path = version_dir / MANIFEST_FILE_NAME
    manifest_path.write_text(json.dumps(manifest_dict, indent=2) + "\n", encoding="utf-8")


def update_catalog(catalog_path: Path, manifest: SnapshotManifest) -> None:
    """Append manifest entry to dataset catalog.

    Args:
        catalog_path: Catalog JSON path.
        manifest: Manifest to append.
    """
    if catalog_path.exists():
        catalog = read_catalog_file(catalog_path)
    else:
        catalog = {"latest_version": None, "versions": []}
    manifest_dict = asdict(manifest)
    manifest_dict["created_at"] = manifest.created_at.isoformat()
    versions = cast(list[dict[str, Any]], catalog["versions"])
    versions.append(manifest_dict)
    catalog["latest_version"] = manifest.version_id
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")


def read_catalog_file(catalog_path: Path) -> dict[str, Any]:
    """Read and validate dataset catalog payload.

    Args:
        catalog_path: Catalog JSON path.

    Returns:
        Parsed catalog object.

    Raises:
        ForgeStoreError: If catalog is missing or invalid.
    """
    if not catalog_path.exists():
        raise ForgeStoreError(
            f"Dataset catalog not found at {catalog_path}. "
            "Ingest data before requesting versions."
        )
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ForgeStoreError(
            f"Failed to parse dataset catalog at {catalog_path}: {error.msg}. "
            "Recreate the dataset catalog from source snapshots."
        ) from error
    if not isinstance(payload, dict):
        raise ForgeStoreError(
            f"Failed to parse dataset catalog at {catalog_path}: "
            "expected JSON object at top level. Recreate the catalog."
        )
    return payload


def manifest_from_dict(payload: dict[str, Any]) -> SnapshotManifest:
    """Deserialize manifest payload from dictionary.

    Args:
        payload: Manifest dictionary.

    Returns:
        Typed snapshot manifest.
    """
    return SnapshotManifest(
        dataset_name=str(payload["dataset_name"]),
        version_id=str(payload["version_id"]),
        created_at=datetime.fromisoformat(str(payload["created_at"])),
        parent_version=str(payload["parent_version"]) if payload["parent_version"] else None,
        recipe_steps=tuple(str(step) for step in payload["recipe_steps"]),
        record_count=int(payload["record_count"]),
        source_uri=payload.get("source_uri"),
    )
