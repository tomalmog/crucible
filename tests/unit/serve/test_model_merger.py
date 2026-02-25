"""Tests for model merger."""

from __future__ import annotations

import pytest

from serve.model_merger import SUPPORTED_MERGE_METHODS, MergeConfig, MergeResult


def test_merge_config_defaults() -> None:
    """MergeConfig has sensible defaults."""
    config = MergeConfig(model_paths=("a.pt", "b.pt"))
    assert config.method == "average"
    assert config.output_path == "./merged_model.pt"


def test_merge_config_frozen() -> None:
    """MergeConfig is immutable."""
    config = MergeConfig(model_paths=("a.pt",))
    with pytest.raises(AttributeError):
        config.method = "slerp"  # type: ignore[misc]


def test_merge_result_creation() -> None:
    """MergeResult stores merge info."""
    result = MergeResult(
        output_path="merged.pt", method="slerp",
        num_models=2, num_parameters=1000000,
    )
    assert result.num_models == 2
    assert result.method == "slerp"


def test_supported_merge_methods() -> None:
    """All expected merge methods are supported."""
    assert "slerp" in SUPPORTED_MERGE_METHODS
    assert "ties" in SUPPORTED_MERGE_METHODS
    assert "dare" in SUPPORTED_MERGE_METHODS
    assert "average" in SUPPORTED_MERGE_METHODS
