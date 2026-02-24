"""Unit tests for model registry type definitions."""

from __future__ import annotations

import pytest

from core.model_registry_types import ModelBranch, ModelTag, ModelVersion


def test_model_version_is_frozen() -> None:
    """ModelVersion should be immutable after construction."""
    version = ModelVersion(
        version_id="mv-abc123",
        model_path="/tmp/model.pt",
        run_id="run-001",
    )
    with pytest.raises(AttributeError):
        version.version_id = "mv-changed"  # type: ignore[misc]


def test_model_version_fields_present() -> None:
    """ModelVersion should expose all expected fields with defaults."""
    version = ModelVersion(
        version_id="mv-abc123",
        model_path="/tmp/model.pt",
        run_id=None,
        tags=("prod", "v1"),
        created_at="2026-01-01T00:00:00+00:00",
        parent_version_id="mv-parent",
    )
    assert version.version_id == "mv-abc123"
    assert version.model_path == "/tmp/model.pt"
    assert version.run_id is None
    assert version.tags == ("prod", "v1")
    assert version.created_at == "2026-01-01T00:00:00+00:00"
    assert version.parent_version_id == "mv-parent"


def test_model_tag_is_frozen() -> None:
    """ModelTag should be immutable after construction."""
    tag = ModelTag(
        tag_name="production",
        version_id="mv-abc123",
        created_at="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(AttributeError):
        tag.tag_name = "staging"  # type: ignore[misc]
    assert tag.tag_name == "production"
    assert tag.version_id == "mv-abc123"
    assert tag.created_at == "2026-01-01T00:00:00+00:00"
