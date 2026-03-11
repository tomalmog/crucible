"""Unit tests for model registry type definitions."""

from __future__ import annotations

import pytest

from core.model_registry_types import DeleteResult, ModelEntry


def test_model_entry_is_frozen() -> None:
    """ModelEntry should be immutable after construction."""
    entry = ModelEntry(
        model_name="test-model",
        model_path="/tmp/model.pt",
        run_id="run-001",
    )
    with pytest.raises(AttributeError):
        entry.model_name = "changed"  # type: ignore[misc]


def test_model_entry_fields_present() -> None:
    """ModelEntry should expose all expected fields with defaults."""
    entry = ModelEntry(
        model_name="my-model",
        model_path="/tmp/model.pt",
        run_id=None,
        created_at="2026-01-01T00:00:00+00:00",
    )
    assert entry.model_name == "my-model"
    assert entry.model_path == "/tmp/model.pt"
    assert entry.run_id is None
    assert entry.created_at == "2026-01-01T00:00:00+00:00"
    assert entry.location_type == "local"
    assert entry.remote_host == ""
    assert entry.remote_path == ""


def test_delete_result_is_frozen() -> None:
    """DeleteResult should be immutable after construction."""
    result = DeleteResult(
        entries_removed=1,
        local_paths_deleted=("/tmp/model.pt",),
        local_paths_skipped=(),
        errors=(),
    )
    with pytest.raises(AttributeError):
        result.entries_removed = 2  # type: ignore[misc]
