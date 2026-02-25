"""End-to-end integration tests for HuggingFace Hub operations.

Tests call the real HuggingFace Hub API — no mocks except for the
missing-dependency error path which legitimately requires one.
"""

from __future__ import annotations

import pytest

from cli.main import main
from core.errors import ForgeDependencyError
from serve.huggingface_hub import (
    HubDatasetInfo,
    HubModelInfo,
    search_datasets,
    search_models,
)


def test_search_models() -> None:
    """search_models should return real HubModelInfo from HuggingFace."""
    results = search_models("llama", limit=3)

    assert len(results) == 3
    for model in results:
        assert isinstance(model, HubModelInfo)
        assert model.repo_id
        assert model.downloads >= 0


def test_search_datasets() -> None:
    """search_datasets should return real HubDatasetInfo from HuggingFace."""
    results = search_datasets("code", limit=2)

    assert len(results) == 2
    for ds in results:
        assert isinstance(ds, HubDatasetInfo)
        assert ds.repo_id


def test_search_models_empty_query() -> None:
    """A nonsense query should not crash — may return empty or partial."""
    results = search_models("xyznonexistent999", limit=5)
    assert isinstance(results, list)


@pytest.mark.skip(reason="downloads large model files")
def test_download_model_skipped() -> None:
    """Placeholder — real download is too slow/large for CI."""


def test_cli_search(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI 'hub search-models bert' should exit 0 and print repo IDs."""
    exit_code = main([
        "--data-root", str(tmp_path),
        "hub", "search-models", "bert", "--limit", "3",
    ])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "bert" in captured.lower() or "downloads=" in captured


def test_missing_dep(monkeypatch: pytest.MonkeyPatch) -> None:
    """search_models should raise ForgeDependencyError when hub is absent."""

    def _raise_dep_error() -> None:
        raise ForgeDependencyError("huggingface_hub is required")

    monkeypatch.setattr(
        "serve.huggingface_hub._import_huggingface_hub",
        _raise_dep_error,
    )

    with pytest.raises(ForgeDependencyError, match="huggingface_hub is required"):
        search_models("test")
