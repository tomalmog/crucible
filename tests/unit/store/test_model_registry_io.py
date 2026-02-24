"""Unit tests for model registry I/O persistence helpers."""

from __future__ import annotations

import pytest

from core.errors import ForgeModelRegistryError
from core.model_registry_types import ModelTag, ModelVersion
from store.model_registry_io import (
    load_model_tag,
    load_model_version,
    load_registry_index,
    save_model_tag,
    save_model_version,
    save_registry_index,
)


def test_save_and_load_model_version(tmp_path) -> None:
    """save_model_version then load_model_version should round-trip."""
    models_root = tmp_path / "models"
    version = ModelVersion(
        version_id="mv-test123456",
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
    assert loaded.model_path == version.model_path
    assert loaded.run_id == version.run_id
    assert loaded.tags == ("prod",)
    assert loaded.created_at == version.created_at
    assert loaded.parent_version_id is None


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


def test_save_and_load_registry_index(tmp_path) -> None:
    """save_registry_index then load_registry_index should round-trip."""
    models_root = tmp_path / "models"
    index = {
        "version_ids": ["mv-aaa", "mv-bbb"],
        "active_version_id": "mv-aaa",
    }
    path = save_registry_index(models_root, index)
    assert path.exists()
    loaded = load_registry_index(models_root)
    assert loaded["version_ids"] == ["mv-aaa", "mv-bbb"]
    assert loaded["active_version_id"] == "mv-aaa"


def test_load_registry_index_returns_default_when_missing(tmp_path) -> None:
    """load_registry_index should return defaults when no index exists."""
    models_root = tmp_path / "models"
    loaded = load_registry_index(models_root)
    assert loaded["version_ids"] == []
    assert loaded["active_version_id"] is None
