"""Unit tests for model version diff utilities."""

from __future__ import annotations

from core.model_registry_types import ModelVersion
from store.model_diff import diff_model_versions, format_model_diff


def test_diff_identical_versions_returns_empty() -> None:
    """Diffing identical versions should return an empty dict."""
    version = ModelVersion(
        version_id="mv-aaa",
        model_path="/tmp/m.pt",
        run_id="r1",
        created_at="2026-01-01T00:00:00+00:00",
    )
    diff = diff_model_versions(version, version)
    assert diff == {}


def test_diff_different_versions_detects_changes() -> None:
    """Diffing versions with differences should return changed fields."""
    v_a = ModelVersion(
        version_id="mv-aaa",
        model_path="/tmp/m1.pt",
        run_id="r1",
        created_at="2026-01-01T00:00:00+00:00",
    )
    v_b = ModelVersion(
        version_id="mv-bbb",
        model_path="/tmp/m2.pt",
        run_id="r2",
        created_at="2026-01-02T00:00:00+00:00",
    )
    diff = diff_model_versions(v_a, v_b)
    assert "version_id" in diff
    assert "model_path" in diff
    assert "run_id" in diff
    assert diff["model_path"] == ("/tmp/m1.pt", "/tmp/m2.pt")


def test_format_model_diff_empty() -> None:
    """format_model_diff on empty diff should report no differences."""
    lines = format_model_diff({})
    assert lines == ("No differences found.",)


def test_format_model_diff_produces_lines() -> None:
    """format_model_diff should produce readable lines for each field."""
    diff = {
        "model_path": ("/tmp/m1.pt", "/tmp/m2.pt"),
        "run_id": ("r1", "r2"),
    }
    lines = format_model_diff(diff)
    assert len(lines) == 2
    assert "model_path" in lines[0]
    assert "run_id" in lines[1]
    assert "->" in lines[0]
