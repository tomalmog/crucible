"""Tests for HuggingFace Hub integration."""

from __future__ import annotations

import pytest

from serve.huggingface_hub import HubDatasetInfo, HubModelInfo


def test_hub_model_info_creation() -> None:
    """HubModelInfo stores model metadata."""
    info = HubModelInfo(
        repo_id="test/model",
        author="test",
        downloads=1000,
        likes=50,
        tags=("text-generation",),
        pipeline_tag="text-generation",
    )
    assert info.repo_id == "test/model"
    assert info.downloads == 1000
    assert info.pipeline_tag == "text-generation"


def test_hub_model_info_defaults() -> None:
    """HubModelInfo has sensible defaults."""
    info = HubModelInfo(repo_id="test/model")
    assert info.author == ""
    assert info.downloads == 0
    assert info.tags == ()


def test_hub_dataset_info_creation() -> None:
    """HubDatasetInfo stores dataset metadata."""
    info = HubDatasetInfo(
        repo_id="test/dataset",
        author="test",
        downloads=500,
        tags=("text",),
    )
    assert info.repo_id == "test/dataset"
    assert info.downloads == 500


def test_hub_model_info_frozen() -> None:
    """HubModelInfo is immutable."""
    info = HubModelInfo(repo_id="test/model")
    with pytest.raises(AttributeError):
        info.repo_id = "other"  # type: ignore[misc]


def test_hub_dataset_info_frozen() -> None:
    """HubDatasetInfo is immutable."""
    info = HubDatasetInfo(repo_id="test/dataset")
    with pytest.raises(AttributeError):
        info.repo_id = "other"  # type: ignore[misc]
