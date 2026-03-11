"""Unit tests for ModelRegistry operations."""

from __future__ import annotations

import pytest

from core.errors import CrucibleModelRegistryError
from store.model_registry import ModelRegistry


def test_register_model_creates_entry(tmp_path) -> None:
    """Registering a model should return an entry with the given name."""
    registry = ModelRegistry(tmp_path)
    entry = registry.register_model("my-model", "/tmp/model.pt", run_id="run-001")
    assert entry.model_name == "my-model"
    assert entry.model_path == "/tmp/model.pt"
    assert entry.run_id == "run-001"
    assert entry.created_at != ""


def test_list_models_returns_registered(tmp_path) -> None:
    """list_models should return all registered entries."""
    registry = ModelRegistry(tmp_path)
    registry.register_model("model-a", "/tmp/m1.pt")
    registry.register_model("model-b", "/tmp/m2.pt")
    models = registry.list_models()
    assert len(models) == 2
    names = {m.model_name for m in models}
    assert names == {"model-a", "model-b"}


def test_register_model_overwrites(tmp_path) -> None:
    """Registering a model with the same name should update the entry."""
    registry = ModelRegistry(tmp_path)
    registry.register_model("my-model", "/tmp/m1.pt")
    registry.register_model("my-model", "/tmp/m2.pt")
    models = registry.list_models()
    assert len(models) == 1
    assert models[0].model_path == "/tmp/m2.pt"


def test_list_model_names(tmp_path) -> None:
    """list_model_names should return all distinct model names."""
    registry = ModelRegistry(tmp_path)
    registry.register_model("alpha", "/tmp/m1.pt")
    registry.register_model("beta", "/tmp/m2.pt")
    names = registry.list_model_names()
    assert names == ("alpha", "beta")


def test_get_model_returns_correct(tmp_path) -> None:
    """get_model should return the specific entry requested."""
    registry = ModelRegistry(tmp_path)
    registry.register_model("my-model", "/tmp/m1.pt", run_id="r1")
    fetched = registry.get_model("my-model")
    assert fetched.model_name == "my-model"
    assert fetched.model_path == "/tmp/m1.pt"
    assert fetched.run_id == "r1"


def test_get_model_raises_for_missing(tmp_path) -> None:
    """get_model should raise CrucibleModelRegistryError for unknown name."""
    registry = ModelRegistry(tmp_path)
    with pytest.raises(CrucibleModelRegistryError, match="not found"):
        registry.get_model("nonexistent")


def test_delete_model_removes_entry(tmp_path) -> None:
    """delete_model should remove the entry from the registry."""
    registry = ModelRegistry(tmp_path)
    registry.register_model("my-model", "/tmp/m1.pt")
    result = registry.delete_model("my-model")
    assert result.entries_removed == 1
    assert registry.list_model_names() == ()


def test_register_remote_model(tmp_path) -> None:
    """register_remote_model should create an entry with remote location."""
    registry = ModelRegistry(tmp_path)
    entry = registry.register_remote_model(
        "remote-model", "hpc.example.com", "/remote/model.pt", run_id="rj-123",
    )
    assert entry.location_type == "remote"
    assert entry.remote_host == "hpc.example.com"
    assert entry.remote_path == "/remote/model.pt"
    assert entry.model_path == ""


def test_mark_model_pulled(tmp_path) -> None:
    """mark_model_pulled should update location to 'both'."""
    registry = ModelRegistry(tmp_path)
    registry.register_remote_model(
        "remote-model", "hpc.example.com", "/remote/model.pt",
    )
    updated = registry.mark_model_pulled("remote-model", "/local/model.pt")
    assert updated.location_type == "both"
    assert updated.model_path == "/local/model.pt"
    assert updated.remote_host == "hpc.example.com"
