"""Unit tests for model registry I/O persistence helpers."""

from __future__ import annotations

import json

import pytest

from core.errors import ForgeModelRegistryError
from core.model_registry_types import ModelTag, ModelVersion
from store.model_registry_io import (
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


def test_save_and_load_model_version(tmp_path) -> None:
    """save_model_version then load_model_version should round-trip."""
    models_root = tmp_path / "models"
    version = ModelVersion(
        version_id="mv-test123456",
        model_name="my-model",
        model_path="/tmp/model.pt",
        run_id="run-001",
        tags=("prod",),
        created_at="2026-01-01T00:00:00+00:00",
        parent_version_id=None,
    )
    path = save_model_version(models_root, version)
    assert path.exists()
    loaded = load_model_version(models_root, "mv-test123456")
    assert loaded.version_id == version.version_id
    assert loaded.model_name == "my-model"
    assert loaded.model_path == version.model_path
    assert loaded.run_id == version.run_id
    assert loaded.tags == ("prod",)
    assert loaded.created_at == version.created_at
    assert loaded.parent_version_id is None


def test_load_model_version_defaults_model_name(tmp_path) -> None:
    """Loading a version without model_name should default to 'default'."""
    models_root = tmp_path / "models"
    versions_dir = models_root / "versions"
    versions_dir.mkdir(parents=True)
    # Write a version file without model_name field (old format)
    data = {
        "version_id": "mv-old123",
        "model_path": "/tmp/old.pt",
        "run_id": None,
        "tags": [],
        "created_at": "2026-01-01T00:00:00+00:00",
        "parent_version_id": None,
    }
    (versions_dir / "mv-old123.json").write_text(json.dumps(data))
    loaded = load_model_version(models_root, "mv-old123")
    assert loaded.model_name == "default"


def test_load_model_version_raises_for_missing(tmp_path) -> None:
    """load_model_version should raise for nonexistent version."""
    models_root = tmp_path / "models"
    models_root.mkdir(parents=True)
    with pytest.raises(ForgeModelRegistryError, match="Failed to load"):
        load_model_version(models_root, "mv-doesnotexist")


def test_save_and_load_model_tag(tmp_path) -> None:
    """save_model_tag then load_model_tag should round-trip."""
    models_root = tmp_path / "models"
    tag = ModelTag(
        tag_name="production",
        version_id="mv-test123456",
        created_at="2026-01-01T00:00:00+00:00",
    )
    path = save_model_tag(models_root, tag)
    assert path.exists()
    loaded = load_model_tag(models_root, "production")
    assert loaded.tag_name == tag.tag_name
    assert loaded.version_id == tag.version_id
    assert loaded.created_at == tag.created_at


def test_load_model_tag_raises_for_missing(tmp_path) -> None:
    """load_model_tag should raise for nonexistent tag."""
    models_root = tmp_path / "models"
    models_root.mkdir(parents=True)
    with pytest.raises(ForgeModelRegistryError, match="Failed to load"):
        load_model_tag(models_root, "nonexistent-tag")


def test_save_and_load_model_group(tmp_path) -> None:
    """save_model_group then load_model_group should round-trip."""
    models_root = tmp_path / "models"
    group_data = {
        "version_ids": ["mv-aaa", "mv-bbb"],
        "active_version_id": "mv-aaa",
    }
    path = save_model_group(models_root, "my-model", group_data)
    assert path.exists()
    loaded = load_model_group(models_root, "my-model")
    assert loaded["version_ids"] == ["mv-aaa", "mv-bbb"]
    assert loaded["active_version_id"] == "mv-aaa"


def test_load_model_group_returns_default_when_missing(tmp_path) -> None:
    """load_model_group should return defaults when no group file exists."""
    models_root = tmp_path / "models"
    loaded = load_model_group(models_root, "nonexistent")
    assert loaded["version_ids"] == []
    assert loaded["active_version_id"] is None


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


def test_migrate_flat_to_grouped(tmp_path) -> None:
    """migrate_flat_to_grouped should convert old flat format to grouped."""
    models_root = tmp_path / "models"
    models_root.mkdir(parents=True)
    # Write old-format index
    old_index = {"version_ids": ["mv-aaa", "mv-bbb"], "active_version_id": "mv-bbb"}
    (models_root / "index.json").write_text(json.dumps(old_index))
    # Write version files without model_name
    versions_dir = models_root / "versions"
    versions_dir.mkdir()
    for vid in ["mv-aaa", "mv-bbb"]:
        data = {
            "version_id": vid,
            "model_path": f"/tmp/{vid}.pt",
            "run_id": None,
            "tags": [],
            "created_at": "2026-01-01T00:00:00+00:00",
            "parent_version_id": None,
        }
        (versions_dir / f"{vid}.json").write_text(json.dumps(data))

    migrate_flat_to_grouped(models_root)

    # Check new index format
    new_index = json.loads((models_root / "index.json").read_text())
    assert "model_names" in new_index
    assert new_index["model_names"] == ["default"]
    assert "version_ids" not in new_index

    # Check group file
    group = json.loads((models_root / "groups" / "default.json").read_text())
    assert group["version_ids"] == ["mv-aaa", "mv-bbb"]
    assert group["active_version_id"] == "mv-bbb"

    # Check versions were backfilled with model_name
    v_data = json.loads((versions_dir / "mv-aaa.json").read_text())
    assert v_data["model_name"] == "default"


def test_migrate_is_idempotent(tmp_path) -> None:
    """Running migration twice should not change already-migrated data."""
    models_root = tmp_path / "models"
    models_root.mkdir(parents=True)
    # Write new-format index
    new_index = {"model_names": ["my-model"]}
    (models_root / "index.json").write_text(json.dumps(new_index))

    migrate_flat_to_grouped(models_root)

    loaded = json.loads((models_root / "index.json").read_text())
    assert loaded == new_index
