"""Persistence helpers for model registry storage.

This module handles reading and writing model version, tag, and index
JSON files under the .forge/models/ directory tree.
"""

from __future__ import annotations

from pathlib import Path

from core.errors import ForgeModelRegistryError
from core.model_registry_types import ModelTag, ModelVersion
from serve.training_run_io import read_json_file, write_json_file


def save_model_version(models_root: Path, version: ModelVersion) -> Path:
    """Persist a model version record to disk.

    Args:
        models_root: Root path for model registry storage.
        version: Model version to save.

    Returns:
        Path to the written JSON file.
    """
    versions_dir = models_root / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    payload = _version_to_dict(version)
    target = versions_dir / f"{version.version_id}.json"
    try:
        write_json_file(target, payload)
    except Exception as error:
        raise ForgeModelRegistryError(
            f"Failed to save model version {version.version_id}: {error}."
        ) from error
    return target


def load_model_version(models_root: Path, version_id: str) -> ModelVersion:
    """Load a model version record from disk.

    Args:
        models_root: Root path for model registry storage.
        version_id: Identifier of the version to load.

    Returns:
        Loaded ModelVersion instance.
    """
    target = models_root / "versions" / f"{version_id}.json"
    try:
        raw = read_json_file(target)
    except Exception as error:
        raise ForgeModelRegistryError(
            f"Failed to load model version {version_id}: {error}."
        ) from error
    if not isinstance(raw, dict):
        raise ForgeModelRegistryError(
            f"Invalid version data for {version_id}."
        )
    return _dict_to_version(raw)


def save_model_tag(models_root: Path, tag: ModelTag) -> Path:
    """Persist a model tag record to disk.

    Args:
        models_root: Root path for model registry storage.
        tag: Model tag to save.

    Returns:
        Path to the written JSON file.
    """
    tags_dir = models_root / "tags"
    tags_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag_name": tag.tag_name,
        "version_id": tag.version_id,
        "created_at": tag.created_at,
    }
    target = tags_dir / f"{tag.tag_name}.json"
    try:
        write_json_file(target, payload)
    except Exception as error:
        raise ForgeModelRegistryError(
            f"Failed to save model tag {tag.tag_name}: {error}."
        ) from error
    return target


def load_model_tag(models_root: Path, tag_name: str) -> ModelTag:
    """Load a model tag record from disk.

    Args:
        models_root: Root path for model registry storage.
        tag_name: Name of the tag to load.

    Returns:
        Loaded ModelTag instance.
    """
    target = models_root / "tags" / f"{tag_name}.json"
    try:
        raw = read_json_file(target)
    except Exception as error:
        raise ForgeModelRegistryError(
            f"Failed to load model tag {tag_name}: {error}."
        ) from error
    if not isinstance(raw, dict):
        raise ForgeModelRegistryError(
            f"Invalid tag data for {tag_name}."
        )
    return ModelTag(
        tag_name=raw["tag_name"],
        version_id=raw["version_id"],
        created_at=raw.get("created_at", ""),
    )


def save_registry_index(models_root: Path, index: dict[str, object]) -> Path:
    """Persist the registry index to disk.

    Args:
        models_root: Root path for model registry storage.
        index: Index dictionary to save.

    Returns:
        Path to the written JSON file.
    """
    models_root.mkdir(parents=True, exist_ok=True)
    target = models_root / "index.json"
    try:
        write_json_file(target, index)
    except Exception as error:
        raise ForgeModelRegistryError(
            f"Failed to save registry index: {error}."
        ) from error
    return target


def load_registry_index(models_root: Path) -> dict[str, object]:
    """Load the registry index from disk.

    Args:
        models_root: Root path for model registry storage.

    Returns:
        Index dictionary with version IDs and active version.
    """
    target = models_root / "index.json"
    default: dict[str, object] = {"version_ids": [], "active_version_id": None}
    try:
        raw = read_json_file(target, default_value=default)
    except Exception as error:
        raise ForgeModelRegistryError(
            f"Failed to load registry index: {error}."
        ) from error
    if not isinstance(raw, dict):
        return dict(default)
    return dict(raw)


def _version_to_dict(version: ModelVersion) -> dict[str, object]:
    """Convert a ModelVersion to a serializable dictionary."""
    return {
        "version_id": version.version_id,
        "model_path": version.model_path,
        "run_id": version.run_id,
        "tags": list(version.tags),
        "created_at": version.created_at,
        "parent_version_id": version.parent_version_id,
    }


def _dict_to_version(raw: dict[str, object]) -> ModelVersion:
    """Reconstruct a ModelVersion from a dictionary."""
    return ModelVersion(
        version_id=str(raw["version_id"]),
        model_path=str(raw["model_path"]),
        run_id=raw.get("run_id") if raw.get("run_id") is not None else None,  # type: ignore[arg-type]
        tags=tuple(raw.get("tags", ())),  # type: ignore[arg-type]
        created_at=str(raw.get("created_at", "")),
        parent_version_id=raw.get("parent_version_id") if raw.get("parent_version_id") is not None else None,  # type: ignore[arg-type]
    )
