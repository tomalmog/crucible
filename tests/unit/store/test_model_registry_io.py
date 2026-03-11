"""Unit tests for model registry I/O persistence helpers."""

from __future__ import annotations

import json

import pytest

from core.errors import CrucibleModelRegistryError
from core.model_registry_types import ModelEntry
from store.model_registry_io import (
    delete_model_entry_file,
    load_model_entry,
    load_registry_index,
    migrate_versioned_to_flat,
    save_model_entry,
    save_registry_index,
)


def test_save_and_load_model_entry(tmp_path) -> None:
    """save_model_entry then load_model_entry should round-trip."""
    models_root = tmp_path / "models"
    entry = ModelEntry(
        model_name="my-model",
        model_path="/tmp/model.pt",
        run_id="run-001",
        created_at="2026-01-01T00:00:00+00:00",
    )
    path = save_model_entry(models_root, entry)
    assert path.exists()
    loaded = load_model_entry(models_root, "my-model")
    assert loaded.model_name == "my-model"
    assert loaded.model_path == entry.model_path
    assert loaded.run_id == entry.run_id
    assert loaded.created_at == entry.created_at


def test_load_model_entry_raises_for_missing(tmp_path) -> None:
    """load_model_entry should raise for nonexistent model."""
    models_root = tmp_path / "models"
    models_root.mkdir(parents=True)
    with pytest.raises(CrucibleModelRegistryError, match="Failed to load"):
        load_model_entry(models_root, "nonexistent")


def test_delete_model_entry_file(tmp_path) -> None:
    """delete_model_entry_file should remove the entry file."""
    models_root = tmp_path / "models"
    entry = ModelEntry(model_name="my-model", model_path="/tmp/model.pt")
    save_model_entry(models_root, entry)
    delete_model_entry_file(models_root, "my-model")
    assert not (models_root / "entries" / "my-model.json").exists()


def test_save_and_load_registry_index(tmp_path) -> None:
    """save_registry_index then load_registry_index should round-trip."""
    models_root = tmp_path / "models"
    index = {"model_names": ["alpha", "beta"]}
    path = save_registry_index(models_root, index)
    assert path.exists()
    loaded = load_registry_index(models_root)
    assert loaded["model_names"] == ["alpha", "beta"]


def test_load_registry_index_returns_default_when_missing(tmp_path) -> None:
    """load_registry_index should return defaults when no index exists."""
    models_root = tmp_path / "models"
    loaded = load_registry_index(models_root)
    assert loaded["model_names"] == []


def test_migrate_versioned_to_flat_grouped_format(tmp_path) -> None:
    """Migration should convert grouped format to flat entries."""
    models_root = tmp_path / "models"
    models_root.mkdir(parents=True)
    # Write grouped-format index
    index = {"model_names": ["my-model"]}
    (models_root / "index.json").write_text(json.dumps(index))
    # Write group file
    groups_dir = models_root / "groups"
    groups_dir.mkdir()
    group = {"version_ids": ["mv-aaa"], "active_version_id": "mv-aaa"}
    (groups_dir / "my-model.json").write_text(json.dumps(group))
    # Write version file
    versions_dir = models_root / "versions"
    versions_dir.mkdir()
    version = {
        "version_id": "mv-aaa",
        "model_name": "my-model",
        "model_path": "/tmp/model.pt",
        "run_id": "r1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "location_type": "local",
    }
    (versions_dir / "mv-aaa.json").write_text(json.dumps(version))

    migrate_versioned_to_flat(models_root)

    # Entry should exist
    entry_path = models_root / "entries" / "my-model.json"
    assert entry_path.exists()
    data = json.loads(entry_path.read_text())
    assert data["model_name"] == "my-model"
    assert data["model_path"] == "/tmp/model.pt"
    assert data["run_id"] == "r1"
    # Old dirs should be removed
    assert not versions_dir.exists()
    assert not groups_dir.exists()


def test_migrate_versioned_to_flat_old_format(tmp_path) -> None:
    """Migration should convert old flat format (version_ids) to entries."""
    models_root = tmp_path / "models"
    models_root.mkdir(parents=True)
    old_index = {"version_ids": ["mv-bbb"], "active_version_id": "mv-bbb"}
    (models_root / "index.json").write_text(json.dumps(old_index))
    versions_dir = models_root / "versions"
    versions_dir.mkdir()
    version = {
        "version_id": "mv-bbb",
        "model_path": "/tmp/old.pt",
        "run_id": None,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    (versions_dir / "mv-bbb.json").write_text(json.dumps(version))

    migrate_versioned_to_flat(models_root)

    entry_path = models_root / "entries" / "default.json"
    assert entry_path.exists()
    data = json.loads(entry_path.read_text())
    assert data["model_name"] == "default"
    assert data["model_path"] == "/tmp/old.pt"
    # Old dirs should be removed
    assert not versions_dir.exists()


def test_migrate_is_idempotent(tmp_path) -> None:
    """Running migration twice should not change already-migrated data."""
    models_root = tmp_path / "models"
    models_root.mkdir(parents=True)
    index = {"model_names": ["my-model"]}
    (models_root / "index.json").write_text(json.dumps(index))
    # Create entry file (already migrated)
    entries_dir = models_root / "entries"
    entries_dir.mkdir()
    entry = {"model_name": "my-model", "model_path": "/tmp/model.pt"}
    (entries_dir / "my-model.json").write_text(json.dumps(entry))

    migrate_versioned_to_flat(models_root)

    loaded = json.loads((entries_dir / "my-model.json").read_text())
    assert loaded == entry


def test_safe_filename_replaces_slash(tmp_path) -> None:
    """Models with slashes in names should be safely stored."""
    models_root = tmp_path / "models"
    entry = ModelEntry(model_name="org/model", model_path="/tmp/m.pt")
    save_model_entry(models_root, entry)
    loaded = load_model_entry(models_root, "org/model")
    assert loaded.model_name == "org/model"
