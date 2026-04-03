"""Flat model registry for Crucible.

This module provides the main ModelRegistry class that manages
model entries under .crucible/models/.
"""

from __future__ import annotations

import fcntl
import logging
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from core.errors import CrucibleModelRegistryError
from core.model_registry_types import DeleteResult, ModelEntry
from store.model_registry_io import (
    delete_model_entry_file,
    load_model_entry,
    load_registry_index,
    migrate_versioned_to_flat,
    save_model_entry,
    save_registry_index,
)


class ModelRegistry:
    """Manages model entries persisted under .crucible/models/.

    Provides flat CRUD for model artifacts produced by training runs.
    """

    def __init__(self, data_root: Path) -> None:
        self._data_root = data_root
        self._models_root = data_root / "models"
        migrate_versioned_to_flat(self._models_root)

    @property
    def models_root(self) -> Path:
        return self._models_root

    def register_model(
        self,
        model_name: str,
        model_path: str,
        run_id: str | None = None,
    ) -> ModelEntry:
        """Register or update a model entry in the registry."""
        entry = ModelEntry(
            model_name=model_name,
            model_path=model_path,
            run_id=run_id,
            created_at=_now_iso(),
        )
        save_model_entry(self._models_root, entry)
        _ensure_name_in_index(self._models_root, model_name)
        return entry

    def register_remote_model(
        self,
        model_name: str,
        remote_host: str,
        remote_path: str,
        run_id: str | None = None,
    ) -> ModelEntry:
        """Register a model that lives on a remote cluster."""
        entry = ModelEntry(
            model_name=model_name,
            model_path="",
            run_id=run_id,
            created_at=_now_iso(),
            location_type="remote",
            remote_host=remote_host,
            remote_path=remote_path,
        )
        save_model_entry(self._models_root, entry)
        _ensure_name_in_index(self._models_root, model_name)
        return entry

    def mark_model_pulled(
        self,
        model_name: str,
        local_path: str,
    ) -> ModelEntry:
        """Mark a remote model as pulled to local storage."""
        from dataclasses import replace

        existing = self.get_model(model_name)
        updated = replace(
            existing, model_path=local_path, location_type="both",
        )
        save_model_entry(self._models_root, updated)
        return updated

    def update_model_location(
        self,
        model_name: str,
        remote_host: str,
        remote_path: str,
    ) -> ModelEntry:
        """Mark a local model as also available on a remote cluster."""
        from dataclasses import replace

        existing = self.get_model(model_name)
        updated = replace(
            existing,
            location_type="both",
            remote_host=remote_host,
            remote_path=remote_path,
        )
        save_model_entry(self._models_root, updated)
        return updated

    def update_model_to_local_only(self, model_name: str) -> ModelEntry:
        """Strip remote info from a model, keeping only the local path."""
        from dataclasses import replace

        existing = self.get_model(model_name)
        updated = replace(
            existing,
            location_type="local",
            remote_host="",
            remote_path="",
        )
        save_model_entry(self._models_root, updated)
        return updated

    def list_models(self) -> tuple[ModelEntry, ...]:
        """List all registered model entries."""
        entries: list[ModelEntry] = []
        for name in self.list_model_names():
            try:
                entries.append(load_model_entry(self._models_root, name))
            except CrucibleModelRegistryError as exc:
                _logger.warning("Skipping corrupt model entry '%s': %s", name, exc)
                continue
        return tuple(entries)

    def list_model_names(self) -> tuple[str, ...]:
        """List all registered model names."""
        index = load_registry_index(self._models_root)
        names = index.get("model_names", [])
        if not isinstance(names, list):
            return ()
        return tuple(str(n) for n in names)

    def get_model(self, model_name: str) -> ModelEntry:
        """Load a specific model entry by name."""
        entries_path = self._models_root / "entries"
        from store.model_registry_io import _safe_filename
        entry_file = entries_path / f"{_safe_filename(model_name)}.json"
        if not entry_file.exists():
            raise CrucibleModelRegistryError(
                f"Model '{model_name}' not found in registry."
            )
        return load_model_entry(self._models_root, model_name)

    def delete_model(
        self,
        model_name: str,
        delete_local: bool = False,
    ) -> DeleteResult:
        """Delete a model entry and optionally its local files."""
        try:
            entry = self.get_model(model_name)
        except CrucibleModelRegistryError:
            return DeleteResult(
                entries_removed=0,
                local_paths_deleted=(),
                local_paths_skipped=(),
                errors=(f"Model '{model_name}' not found.",),
            )

        deleted: list[str] = []
        skipped: list[str] = []
        errors: list[str] = []

        if delete_local and entry.model_path:
            ok, reason = _safe_delete_local_path(
                self._data_root, entry.model_path,
            )
            if ok:
                deleted.append(entry.model_path)
            else:
                skipped.append(entry.model_path)
                if reason:
                    errors.append(reason)

        try:
            delete_model_entry_file(self._models_root, model_name)
        except Exception as exc:
            errors.append(f"Failed to remove entry file {model_name}: {exc}")

        _remove_name_from_index(self._models_root, model_name)

        return DeleteResult(
            entries_removed=1,
            local_paths_deleted=tuple(deleted),
            local_paths_skipped=tuple(skipped),
            errors=tuple(errors),
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _index_lock(models_root: Path):
    """File lock around registry index read-modify-write operations."""
    lock_path = models_root / ".index.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def _ensure_name_in_index(models_root: Path, model_name: str) -> None:
    with _index_lock(models_root):
        index = load_registry_index(models_root)
        model_names = list(index.get("model_names", []))
        if model_name not in model_names:
            model_names.append(model_name)
            index["model_names"] = model_names
        # Always re-save to bump mtime so the UI detects changes
        save_registry_index(models_root, index)


_logger = logging.getLogger(__name__)

def _safe_delete_local_path(
    data_root: Path,
    model_path: str,
) -> tuple[bool, str]:
    """Delete a local model path."""
    try:
        resolved = Path(model_path).resolve()
    except (OSError, ValueError) as exc:
        return False, f"Cannot resolve path {model_path}: {exc}"

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
    with _index_lock(models_root):
        index = load_registry_index(models_root)
        model_names = list(index.get("model_names", []))
        if model_name in model_names:
            model_names.remove(model_name)
            index["model_names"] = model_names
            save_registry_index(models_root, index)
