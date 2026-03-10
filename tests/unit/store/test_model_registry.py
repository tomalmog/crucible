"""Unit tests for ModelRegistry operations."""

from __future__ import annotations

import pytest

from core.errors import CrucibleModelRegistryError
from store.model_registry import ModelRegistry


def test_register_model_creates_version(tmp_path) -> None:
    """Registering a model should return a version with mv- prefix."""
    registry = ModelRegistry(tmp_path)
    version = registry.register_model("my-model", "/tmp/model.pt", run_id="run-001")
    assert version.version_id.startswith("mv-")
    assert version.model_name == "my-model"
    assert version.model_path == "/tmp/model.pt"
    assert version.run_id == "run-001"
    assert version.created_at != ""


def test_list_versions_returns_registered(tmp_path) -> None:
    """list_versions should return all registered versions across models."""
    registry = ModelRegistry(tmp_path)
    v1 = registry.register_model("model-a", "/tmp/m1.pt")
    v2 = registry.register_model("model-b", "/tmp/m2.pt")
    versions = registry.list_versions()
    assert len(versions) == 2
    ids = {v.version_id for v in versions}
    assert v1.version_id in ids
    assert v2.version_id in ids


def test_list_versions_for_model(tmp_path) -> None:
    """list_versions_for_model should return only versions for that model."""
    registry = ModelRegistry(tmp_path)
    v1 = registry.register_model("model-a", "/tmp/m1.pt")
    v2 = registry.register_model("model-a", "/tmp/m2.pt", parent_version_id=v1.version_id)
    registry.register_model("model-b", "/tmp/m3.pt")
    versions = registry.list_versions_for_model("model-a")
    assert len(versions) == 2
    assert versions[0].version_id == v1.version_id
    assert versions[1].version_id == v2.version_id
    assert versions[1].parent_version_id == v1.version_id


def test_list_model_names(tmp_path) -> None:
    """list_model_names should return all distinct model names."""
    registry = ModelRegistry(tmp_path)
    registry.register_model("alpha", "/tmp/m1.pt")
    registry.register_model("beta", "/tmp/m2.pt")
    registry.register_model("alpha", "/tmp/m3.pt")
    names = registry.list_model_names()
    assert names == ("alpha", "beta")


def test_get_version_returns_correct(tmp_path) -> None:
    """get_version should return the specific version requested."""
    registry = ModelRegistry(tmp_path)
    v1 = registry.register_model("my-model", "/tmp/m1.pt", run_id="r1")
    fetched = registry.get_version(v1.version_id)
    assert fetched.version_id == v1.version_id
    assert fetched.model_path == "/tmp/m1.pt"
    assert fetched.run_id == "r1"


def test_get_version_raises_for_missing(tmp_path) -> None:
    """get_version should raise CrucibleModelRegistryError for unknown ID."""
    registry = ModelRegistry(tmp_path)
    with pytest.raises(CrucibleModelRegistryError, match="not found"):
        registry.get_version("mv-nonexistent")


def test_tag_version_creates_tag(tmp_path) -> None:
    """tag_version should create a named tag pointing to a version."""
    registry = ModelRegistry(tmp_path)
    v1 = registry.register_model("my-model", "/tmp/m1.pt")
    tag = registry.tag_version(v1.version_id, "production")
    assert tag.tag_name == "production"
    assert tag.version_id == v1.version_id
    assert tag.created_at != ""


def test_list_tags_returns_all(tmp_path) -> None:
    """list_tags should return all created tags."""
    registry = ModelRegistry(tmp_path)
    v1 = registry.register_model("my-model", "/tmp/m1.pt")
    registry.tag_version(v1.version_id, "alpha")
    registry.tag_version(v1.version_id, "beta")
    tags = registry.list_tags()
    tag_names = {t.tag_name for t in tags}
    assert tag_names == {"alpha", "beta"}


def test_diff_versions_detects_differences(tmp_path) -> None:
    """diff_versions should detect differing fields between versions."""
    registry = ModelRegistry(tmp_path)
    v1 = registry.register_model("my-model", "/tmp/m1.pt", run_id="r1")
    v2 = registry.register_model("my-model", "/tmp/m2.pt", run_id="r2")
    diff = registry.diff_versions(v1.version_id, v2.version_id)
    assert "model_path" in diff
    assert "run_id" in diff
    assert diff["model_path"] == ("/tmp/m1.pt", "/tmp/m2.pt")


def test_rollback_sets_active_version(tmp_path) -> None:
    """rollback_to_version should update the active version in model group."""
    registry = ModelRegistry(tmp_path)
    v1 = registry.register_model("my-model", "/tmp/m1.pt")
    v2 = registry.register_model("my-model", "/tmp/m2.pt")
    assert registry.get_active_version_id_for_model("my-model") == v1.version_id
    rolled = registry.rollback_to_version("my-model", v2.version_id)
    assert rolled.version_id == v2.version_id
    assert registry.get_active_version_id_for_model("my-model") == v2.version_id


def test_rollback_scoped_to_model(tmp_path) -> None:
    """Rollback in one model should not affect another model's active version."""
    registry = ModelRegistry(tmp_path)
    a1 = registry.register_model("model-a", "/tmp/a1.pt")
    a2 = registry.register_model("model-a", "/tmp/a2.pt")
    b1 = registry.register_model("model-b", "/tmp/b1.pt")
    # model-a active is a1, model-b active is b1
    assert registry.get_active_version_id_for_model("model-a") == a1.version_id
    assert registry.get_active_version_id_for_model("model-b") == b1.version_id
    # Rollback model-a to a2
    registry.rollback_to_version("model-a", a2.version_id)
    assert registry.get_active_version_id_for_model("model-a") == a2.version_id
    # model-b should be unchanged
    assert registry.get_active_version_id_for_model("model-b") == b1.version_id
