"""Model versioning registry for Crucible.

This module provides the main ModelRegistry class that manages
model versions, tags, and rollback under .crucible/models/.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.errors import CrucibleModelRegistryError
from core.model_registry_types import DeleteResult, ModelTag, ModelVersion
from store.model_diff import diff_model_versions
from store.model_registry_io import (
    delete_model_group_file,
    delete_model_version_file,
    load_model_group,
    load_model_tag,
    load_model_version,
    load_registry_index,
    migrate_flat_to_grouped,
    save_model_group,
    save_model_tag,
    save_model_version,
    save_registry_index,
)
from store.model_rollback import rollback_active_version


class ModelRegistry:
    """Manages model versions persisted under .crucible/models/.

    Provides registration, tagging, diffing, and rollback of
    model artifacts produced by training runs.
    """

    def __init__(self, data_root: Path) -> None:
        """Create a model registry rooted at data_root.

        Args:
            data_root: Root .crucible directory for storage.
        """
        self._data_root = data_root
        self._models_root = data_root / "models"
        _ensure_migrated(self._models_root)

    @property
    def models_root(self) -> Path:
        """Return the models storage root path."""
        return self._models_root

    def register_model(
        self,
        model_name: str,
        model_path: str,
        run_id: str | None = None,
        parent_version_id: str | None = None,
    ) -> ModelVersion:
        """Register a new model version in the registry.

        Args:
            model_name: Name of the model to register under.
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
            model_name=model_name,
            model_path=model_path,
            run_id=run_id,
            created_at=created_at,
            parent_version_id=parent_version_id,
        )
        save_model_version(self._models_root, version)
        _append_version_to_group(self._models_root, model_name, version_id)
        return version

    def list_model_names(self) -> tuple[str, ...]:
        """List all registered model names.

        Returns:
            Tuple of model names in registration order.
        """
        index = load_registry_index(self._models_root)
        names = index.get("model_names", [])
        if not isinstance(names, list):
            return ()
        return tuple(str(n) for n in names)

    def list_versions_for_model(self, model_name: str) -> tuple[ModelVersion, ...]:
        """List all versions belonging to a named model.

        Args:
            model_name: Name of the model to list versions for.

        Returns:
            Tuple of ModelVersion records in registration order.
        """
        group = load_model_group(self._models_root, model_name)
        version_ids = group.get("version_ids", [])
        if not isinstance(version_ids, list):
            return ()
        versions: list[ModelVersion] = []
        for vid in version_ids:
            versions.append(load_model_version(self._models_root, str(vid)))
        return tuple(versions)

    def list_versions(self) -> tuple[ModelVersion, ...]:
        """List all registered model versions across all models.

        Returns:
            Tuple of all ModelVersion records.
        """
        all_versions: list[ModelVersion] = []
        for name in self.list_model_names():
            all_versions.extend(self.list_versions_for_model(name))
        return tuple(all_versions)

    def get_version(self, version_id: str) -> ModelVersion:
        """Load a specific model version by ID.

        Args:
            version_id: Identifier of the version to load.

        Returns:
            The requested ModelVersion record.

        Raises:
            CrucibleModelRegistryError: If version does not exist.
        """
        version_path = self._models_root / "versions" / f"{version_id}.json"
        if not version_path.exists():
            raise CrucibleModelRegistryError(
                f"Model version {version_id} not found in registry."
            )
        return load_model_version(self._models_root, version_id)

    def get_active_version_id_for_model(self, model_name: str) -> str | None:
        """Return the active version ID for a named model, or None."""
        group = load_model_group(self._models_root, model_name)
        active = group.get("active_version_id")
        if active is None:
            return None
        return str(active)

    def tag_version(self, version_id: str, tag_name: str) -> ModelTag:
        """Create a named tag pointing to a model version.

        Args:
            version_id: Target version identifier.
            tag_name: Human-readable tag name.

        Returns:
            Newly created ModelTag record.

        Raises:
            CrucibleModelRegistryError: If version does not exist.
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
            CrucibleModelRegistryError: If tag does not exist.
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

    def rollback_to_version(self, model_name: str, version_id: str) -> ModelVersion:
        """Mark a version as the active model version within a model group.

        Args:
            model_name: Name of the model to rollback within.
            version_id: Identifier of the version to activate.

        Returns:
            The ModelVersion now marked as active.

        Raises:
            CrucibleModelRegistryError: If version does not exist.
        """
        return rollback_active_version(self, model_name, version_id)

    def register_remote_model(
        self,
        model_name: str,
        remote_host: str,
        remote_path: str,
        run_id: str | None = None,
    ) -> ModelVersion:
        """Register a model that lives on a remote cluster.

        Args:
            model_name: Name of the model to register under.
            remote_host: Hostname of the remote cluster.
            remote_path: Path to the model on the remote.
            run_id: Optional training run or job ID.

        Returns:
            Newly created ModelVersion record.
        """
        version_id = _generate_version_id()
        created_at = _now_iso()
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            model_path="",
            run_id=run_id,
            created_at=created_at,
            location_type="remote",
            remote_host=remote_host,
            remote_path=remote_path,
        )
        save_model_version(self._models_root, version)
        _append_version_to_group(self._models_root, model_name, version_id)
        return version

    def mark_model_pulled(
        self,
        version_id: str,
        local_path: str,
    ) -> ModelVersion:
        """Mark a remote model as pulled to local storage.

        Args:
            version_id: Version to update.
            local_path: Local filesystem path to the pulled model.

        Returns:
            Updated ModelVersion record.
        """
        from dataclasses import replace
        version = self.get_version(version_id)
        updated = replace(
            version, model_path=local_path, location_type="both",
        )
        save_model_version(self._models_root, updated)
        return updated

    def delete_model(
        self,
        model_name: str,
        delete_local: bool = False,
    ) -> DeleteResult:
        """Delete all versions of a model and the group itself.

        Args:
            model_name: Name of the model group to delete.
            delete_local: Whether to delete local model files.

        Returns:
            Summary of what was deleted.
        """
        versions = self.list_versions_for_model(model_name)
        deleted: list[str] = []
        skipped: list[str] = []
        errors: list[str] = []

        for v in versions:
            if delete_local and v.model_path:
                ok, reason = _safe_delete_local_path(
                    self._data_root, v.model_path,
                )
                if ok:
                    deleted.append(v.model_path)
                else:
                    skipped.append(v.model_path)
                    if reason:
                        errors.append(reason)
            try:
                delete_model_version_file(self._models_root, v.version_id)
            except Exception as exc:
                errors.append(f"Failed to remove version file {v.version_id}: {exc}")

        try:
            delete_model_group_file(self._models_root, model_name)
        except Exception as exc:
            errors.append(f"Failed to remove group file {model_name}: {exc}")

        _remove_name_from_index(self._models_root, model_name)

        return DeleteResult(
            versions_removed=len(versions),
            local_paths_deleted=tuple(deleted),
            local_paths_skipped=tuple(skipped),
            errors=tuple(errors),
        )

    def delete_version(
        self,
        model_name: str,
        version_id: str,
        delete_local: bool = False,
    ) -> DeleteResult:
        """Delete a single version from a model group.

        Args:
            model_name: Name of the model group.
            version_id: Version to delete.
            delete_local: Whether to delete local model files.

        Returns:
            Summary of what was deleted.
        """
        version = self.get_version(version_id)
        deleted: list[str] = []
        skipped: list[str] = []
        errors: list[str] = []

        if delete_local and version.model_path:
            ok, reason = _safe_delete_local_path(
                self._data_root, version.model_path,
            )
            if ok:
                deleted.append(version.model_path)
            else:
                skipped.append(version.model_path)
                if reason:
                    errors.append(reason)

        try:
            delete_model_version_file(self._models_root, version_id)
        except Exception as exc:
            errors.append(f"Failed to remove version file {version_id}: {exc}")

        group = load_model_group(self._models_root, model_name)
        version_ids = [
            vid for vid in group.get("version_ids", []) if vid != version_id
        ]
        if not version_ids:
            try:
                delete_model_group_file(self._models_root, model_name)
            except Exception as exc:
                errors.append(f"Failed to remove group file: {exc}")
            _remove_name_from_index(self._models_root, model_name)
        else:
            active = group.get("active_version_id")
            if active == version_id:
                active = version_ids[-1]
            group["version_ids"] = version_ids
            group["active_version_id"] = active
            save_model_group(self._models_root, model_name, group)

        return DeleteResult(
            versions_removed=1,
            local_paths_deleted=tuple(deleted),
            local_paths_skipped=tuple(skipped),
            errors=tuple(errors),
        )

    def get_active_version_id(self) -> str | None:
        """Return the currently active version ID of the first model, or None.

        Deprecated: use get_active_version_id_for_model instead.
        """
        names = self.list_model_names()
        if not names:
            return None
        return self.get_active_version_id_for_model(names[0])


def _generate_version_id() -> str:
    """Generate a unique model version identifier."""
    return f"mv-{uuid.uuid4().hex[:12]}"


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _ensure_migrated(models_root: Path) -> None:
    """Run migration if the registry uses the old flat format."""
    migrate_flat_to_grouped(models_root)


def _append_version_to_group(
    models_root: Path,
    model_name: str,
    version_id: str,
) -> None:
    """Add a version ID to a model group, creating the group if needed.

    Also ensures the model name is in the global index.

    Args:
        models_root: Root path for model registry storage.
        model_name: Name of the model group.
        version_id: Identifier to append.
    """
    # Update group index
    group = load_model_group(models_root, model_name)
    version_ids = list(group.get("version_ids", []))
    version_ids.append(version_id)
    group["version_ids"] = version_ids
    if group.get("active_version_id") is None:
        group["active_version_id"] = version_id
    save_model_group(models_root, model_name, group)

    # Ensure model name is in global index
    index = load_registry_index(models_root)
    model_names = list(index.get("model_names", []))
    if model_name not in model_names:
        model_names.append(model_name)
        index["model_names"] = model_names
        save_registry_index(models_root, index)


_logger = logging.getLogger(__name__)

_SAFE_SUBDIRS = ("pulled-models", "runs")


def _safe_delete_local_path(
    data_root: Path,
    model_path: str,
) -> tuple[bool, str]:
    """Delete a local model path only if it lives under a safe subdirectory.

    Args:
        data_root: The .crucible root directory.
        model_path: Path to the model artifact.

    Returns:
        Tuple of (deleted, reason). reason is empty on success.
    """
    try:
        resolved = Path(model_path).resolve()
    except (OSError, ValueError) as exc:
        return False, f"Cannot resolve path {model_path}: {exc}"

    safe = False
    for subdir in _SAFE_SUBDIRS:
        safe_root = (data_root / subdir).resolve()
        try:
            resolved.relative_to(safe_root)
            safe = True
            break
        except ValueError:
            continue

    if not safe:
        msg = f"Skipped {model_path}: not under {'/'.join(_SAFE_SUBDIRS)}"
        _logger.warning(msg)
        return False, msg

    if not resolved.exists():
        return False, f"Path does not exist: {model_path}"

    try:
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()
    except OSError as exc:
        return False, f"Failed to delete {model_path}: {exc}"

    return True, ""


def _remove_name_from_index(models_root: Path, model_name: str) -> None:
    """Remove a model name from the global registry index."""
    index = load_registry_index(models_root)
    model_names = list(index.get("model_names", []))
    if model_name in model_names:
        model_names.remove(model_name)
        index["model_names"] = model_names
        save_registry_index(models_root, index)
