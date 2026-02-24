"""Model versioning registry for Forge.

This module provides the main ModelRegistry class that manages
model versions, tags, and rollback under .forge/models/.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.errors import ForgeModelRegistryError
from core.model_registry_types import ModelTag, ModelVersion
from store.model_diff import diff_model_versions
from store.model_registry_io import (
    load_model_tag,
    load_model_version,
    load_registry_index,
    save_model_tag,
    save_model_version,
    save_registry_index,
)
from store.model_rollback import rollback_active_version


class ModelRegistry:
    """Manages model versions persisted under .forge/models/.

    Provides registration, tagging, diffing, and rollback of
    model artifacts produced by training runs.
    """

    def __init__(self, data_root: Path) -> None:
        """Create a model registry rooted at data_root.

        Args:
            data_root: Root .forge directory for storage.
        """
        self._data_root = data_root
        self._models_root = data_root / "models"

    @property
    def models_root(self) -> Path:
        """Return the models storage root path."""
        return self._models_root

    def register_model(
        self,
        model_path: str,
        run_id: str | None = None,
        parent_version_id: str | None = None,
    ) -> ModelVersion:
        """Register a new model version in the registry.

        Args:
            model_path: Filesystem path to the model artifact.
            run_id: Optional training run that produced the model.
            parent_version_id: Optional parent version identifier.

        Returns:
            Newly created ModelVersion record.
        """
        version_id = _generate_version_id()
        created_at = _now_iso()
        version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            run_id=run_id,
            created_at=created_at,
            parent_version_id=parent_version_id,
        )
        save_model_version(self._models_root, version)
        _append_version_to_index(self._models_root, version_id)
        return version

    def list_versions(self) -> tuple[ModelVersion, ...]:
        """List all registered model versions.

        Returns:
            Tuple of all ModelVersion records in registration order.
        """
        index = load_registry_index(self._models_root)
        version_ids = index.get("version_ids", [])
        if not isinstance(version_ids, list):
            return ()
        versions: list[ModelVersion] = []
        for vid in version_ids:
            versions.append(load_model_version(self._models_root, str(vid)))
        return tuple(versions)

    def get_version(self, version_id: str) -> ModelVersion:
        """Load a specific model version by ID.

        Args:
            version_id: Identifier of the version to load.

        Returns:
            The requested ModelVersion record.

        Raises:
            ForgeModelRegistryError: If version does not exist.
        """
        index = load_registry_index(self._models_root)
        version_ids = index.get("version_ids", [])
        if version_id not in version_ids:
            raise ForgeModelRegistryError(
                f"Model version {version_id} not found in registry."
            )
        return load_model_version(self._models_root, version_id)

    def tag_version(self, version_id: str, tag_name: str) -> ModelTag:
        """Create a named tag pointing to a model version.

        Args:
            version_id: Target version identifier.
            tag_name: Human-readable tag name.

        Returns:
            Newly created ModelTag record.

        Raises:
            ForgeModelRegistryError: If version does not exist.
        """
        self.get_version(version_id)
        tag = ModelTag(
            tag_name=tag_name,
            version_id=version_id,
            created_at=_now_iso(),
        )
        save_model_tag(self._models_root, tag)
        return tag

    def list_tags(self) -> tuple[ModelTag, ...]:
        """List all registered model tags.

        Returns:
            Tuple of all ModelTag records.
        """
        tags_dir = self._models_root / "tags"
        if not tags_dir.exists():
            return ()
        tags: list[ModelTag] = []
        for tag_file in sorted(tags_dir.glob("*.json")):
            tag_name = tag_file.stem
            tags.append(load_model_tag(self._models_root, tag_name))
        return tuple(tags)

    def get_version_by_tag(self, tag_name: str) -> ModelVersion:
        """Load the model version pointed to by a tag.

        Args:
            tag_name: Name of the tag to resolve.

        Returns:
            The ModelVersion the tag points to.

        Raises:
            ForgeModelRegistryError: If tag does not exist.
        """
        tag = load_model_tag(self._models_root, tag_name)
        return self.get_version(tag.version_id)

    def diff_versions(
        self,
        version_id_a: str,
        version_id_b: str,
    ) -> dict[str, tuple[object, object]]:
        """Compare two model versions and return differences.

        Args:
            version_id_a: First version to compare.
            version_id_b: Second version to compare.

        Returns:
            Mapping of differing field names to value pairs.
        """
        version_a = self.get_version(version_id_a)
        version_b = self.get_version(version_id_b)
        return diff_model_versions(version_a, version_b)

    def rollback_to_version(self, version_id: str) -> ModelVersion:
        """Mark a version as the active model version.

        Args:
            version_id: Identifier of the version to activate.

        Returns:
            The ModelVersion now marked as active.

        Raises:
            ForgeModelRegistryError: If version does not exist.
        """
        return rollback_active_version(self, version_id)

    def get_active_version_id(self) -> str | None:
        """Return the currently active version ID, or None."""
        index = load_registry_index(self._models_root)
        active = index.get("active_version_id")
        if active is None:
            return None
        return str(active)


def _generate_version_id() -> str:
    """Generate a unique model version identifier."""
    return f"mv-{uuid.uuid4().hex[:12]}"


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _append_version_to_index(models_root: Path, version_id: str) -> None:
    """Add a version ID to the registry index.

    Args:
        models_root: Root path for model registry storage.
        version_id: Identifier to append.
    """
    index = load_registry_index(models_root)
    version_ids = list(index.get("version_ids", []))
    version_ids.append(version_id)
    index["version_ids"] = version_ids
    if index.get("active_version_id") is None:
        index["active_version_id"] = version_id
    save_registry_index(models_root, index)
