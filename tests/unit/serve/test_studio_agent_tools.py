"""Tests for Studio agent tool exposure."""

from __future__ import annotations

from serve.studio_agent import _TOOL_NAMES


def test_agent_exposes_model_health_check_tool() -> None:
    assert "run_model_health_check" in _TOOL_NAMES


def test_agent_exposes_remote_model_health_check_tool() -> None:
    assert "submit_remote_model_health_check" in _TOOL_NAMES


def test_agent_exposes_hub_dataset_search_tool() -> None:
    assert "hub_search_datasets" in _TOOL_NAMES


def test_agent_exposes_hub_dataset_download_tool() -> None:
    assert "hub_download_dataset" in _TOOL_NAMES
