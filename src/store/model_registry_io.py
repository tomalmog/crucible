"""Persistence helpers for model registry storage.

This module handles reading and writing model entry and index JSON
files under the .crucible/models/ directory tree.
"""

from __future__ import annotations

from pathlib import Path

from core.constants import sanitize_remote_name
from core.errors import CrucibleModelRegistryError
from core.model_registry_types import ModelEntry
from serve.training_run_io import read_json_file, write_json_file


def _safe_filename(name: str) -> str:
    """Sanitize a model name for use as a filename.

    Uses ``sanitize_remote_name`` so HuggingFace repo IDs like
    ``org/model`` become ``org_model`` consistently across all paths.
    """
    return sanitize_remote_name(name)


def save_model_entry(models_root: Path, entry: ModelEntry) -> Path:
    """Persist a model entry record to disk.

    Args:
        models_root: Root path for model registry storage.
        entry: Model entry to save.

    Returns:
        Path to the written JSON file.
    """
    entries_dir = models_root / "entries"
    entries_dir.mkdir(parents=True, exist_ok=True)
    payload = _entry_to_dict(entry)
    target = entries_dir / f"{_safe_filename(entry.model_name)}.json"
    try:
        write_json_file(target, payload)
    except Exception as error:
        raise CrucibleModelRegistryError(
            f"Failed to save model entry {entry.model_name}: {error}."
        ) from error
    return target


def load_model_entry(models_root: Path, model_name: str) -> ModelEntry:
    """Load a model entry record from disk.

    Args:
        models_root: Root path for model registry storage.
        model_name: Name of the model to load.

    Returns:
        Loaded ModelEntry instance.
    """
    target = models_root / "entries" / f"{_safe_filename(model_name)}.json"
    try:
        raw = read_json_file(target)
    except Exception as error:
        raise CrucibleModelRegistryError(
            f"Failed to load model entry {model_name}: {error}."
        ) from error
    if not isinstance(raw, dict):
        raise CrucibleModelRegistryError(
            f"Invalid entry data for {model_name}."
        )
    return _dict_to_entry(raw)


def delete_model_entry_file(models_root: Path, model_name: str) -> None:
    """Remove a model entry JSON file from disk.

    Args:
        models_root: Root path for model registry storage.
        model_name: Name of the model to delete.
    """
    target = models_root / "entries" / f"{_safe_filename(model_name)}.json"
    if target.exists():
        target.unlink()


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
        raise CrucibleModelRegistryError(
            f"Failed to save registry index: {error}."
        ) from error
    return target


def load_registry_index(models_root: Path) -> dict[str, object]:
    """Load the registry index from disk.

    Args:
        models_root: Root path for model registry storage.

    Returns:
        Index dictionary with model_names list.
    """
    target = models_root / "index.json"
    default: dict[str, object] = {"model_names": []}
    try:
        raw = read_json_file(target, default_value=default)
    except Exception as error:
        raise CrucibleModelRegistryError(
            f"Failed to load registry index: {error}."
        ) from error
    if not isinstance(raw, dict):
        return dict(default)
    return dict(raw)


def migrate_versioned_to_flat(models_root: Path) -> None:
    """One-time migration from versioned/grouped format to flat entries.

    For each model name in the old index, reads the group's active
    version (or the last version) and creates a flat entry from it.
    Removes old ``groups/``, ``versions/``, and ``tags/`` directories.
    Also handles the oldest flat format (``version_ids`` in index).
    """
    import shutil

    index_path = models_root / "index.json"
    if not index_path.exists():
        return

    raw = read_json_file(index_path, default_value={})
    if not isinstance(raw, dict):
        return

    # Check if entries/ dir already exists — already migrated
    entries_dir = models_root / "entries"
    if entries_dir.is_dir() and any(entries_dir.glob("*.json")):
        return

    versions_dir = models_root / "versions"
    groups_dir = models_root / "groups"
    tags_dir = models_root / "tags"

    # New grouped format: model_names key
    if "model_names" in raw:
        model_names = list(raw.get("model_names", []))
        entries_dir.mkdir(parents=True, exist_ok=True)
        for name in model_names:
            group_path = groups_dir / f"{_safe_filename(str(name))}.json"
            active_vid = None
            version_ids: list[str] = []
            if group_path.exists():
                group_data = read_json_file(group_path, default_value={})
                if isinstance(group_data, dict):
                    active_vid = group_data.get("active_version_id")
                    version_ids = list(group_data.get("version_ids", []))

            # Pick active version, or last, or first
            vid = active_vid or (version_ids[-1] if version_ids else None)
            if vid and versions_dir.is_dir():
                vpath = versions_dir / f"{vid}.json"
                if vpath.exists():
                    vdata = read_json_file(vpath, default_value={})
                    if isinstance(vdata, dict):
                        entry = ModelEntry(
                            model_name=str(name),
                            model_path=str(vdata.get("model_path", "")),
                            run_id=vdata.get("run_id") if vdata.get("run_id") is not None else None,
                            created_at=str(vdata.get("created_at", "")),
                            location_type=str(vdata.get("location_type", "local")),
                            remote_host=str(vdata.get("remote_host", "")),
                            remote_path=str(vdata.get("remote_path", "")),
                        )
                        save_model_entry(models_root, entry)
                        continue
            # No version data found — create stub entry
            entry = ModelEntry(model_name=str(name), model_path="")
            save_model_entry(models_root, entry)

    # Old flat format: version_ids key
    elif "version_ids" in raw:
        version_ids_raw = list(raw.get("version_ids", []))
        active_vid = raw.get("active_version_id")
        vid = active_vid or (version_ids_raw[-1] if version_ids_raw else None)
        if vid and versions_dir.is_dir():
            vpath = versions_dir / f"{vid}.json"
            if vpath.exists():
                vdata = read_json_file(vpath, default_value={})
                if isinstance(vdata, dict):
                    entries_dir.mkdir(parents=True, exist_ok=True)
                    entry = ModelEntry(
                        model_name="default",
                        model_path=str(vdata.get("model_path", "")),
                        run_id=vdata.get("run_id") if vdata.get("run_id") is not None else None,
                        created_at=str(vdata.get("created_at", "")),
                        location_type=str(vdata.get("location_type", "local")),
                        remote_host=str(vdata.get("remote_host", "")),
                        remote_path=str(vdata.get("remote_path", "")),
                    )
                    save_model_entry(models_root, entry)
                    model_names = ["default"]
                    new_index: dict[str, object] = {"model_names": model_names}
                    save_registry_index(models_root, new_index)
    else:
        return

    # Clean up old directories
    for old_dir in (versions_dir, groups_dir, tags_dir):
        if old_dir.is_dir():
            shutil.rmtree(old_dir)


def _entry_to_dict(entry: ModelEntry) -> dict[str, object]:
    """Convert a ModelEntry to a serializable dictionary."""
    return {
        "model_name": entry.model_name,
        "model_path": entry.model_path,
        "run_id": entry.run_id,
        "created_at": entry.created_at,
        "location_type": entry.location_type,
        "remote_host": entry.remote_host,
        "remote_path": entry.remote_path,
    }


def _dict_to_entry(raw: dict[str, object]) -> ModelEntry:
    """Reconstruct a ModelEntry from a dictionary."""
    return ModelEntry(
        model_name=str(raw.get("model_name", "default")),
        model_path=str(raw.get("model_path", "")),
        run_id=str(raw["run_id"]) if raw.get("run_id") is not None else None,
        created_at=str(raw.get("created_at", "")),
        location_type=str(raw.get("location_type", "local")),
        remote_host=str(raw.get("remote_host", "")),
        remote_path=str(raw.get("remote_path", "")),
    )
