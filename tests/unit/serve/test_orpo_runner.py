"""Tests for ORPO training runner."""

from __future__ import annotations

import pytest

from core.orpo_types import OrpoExample, OrpoOptions


def test_orpo_options_defaults() -> None:
    """OrpoOptions defaults are set correctly."""
    opts = OrpoOptions(
        dataset_name="test", output_dir="./out",
        orpo_data_path="data.jsonl",
    )
    assert opts.lambda_orpo == 1.0
    assert opts.beta == 0.1
    assert opts.epochs == 3


def test_orpo_options_custom() -> None:
    """OrpoOptions accepts custom lambda and beta."""
    opts = OrpoOptions(
        dataset_name="test", output_dir="./out",
        orpo_data_path="data.jsonl",
        lambda_orpo=0.5, beta=0.2,
    )
    assert opts.lambda_orpo == 0.5
    assert opts.beta == 0.2


def test_orpo_options_frozen() -> None:
    """OrpoOptions is immutable."""
    opts = OrpoOptions(
        dataset_name="test", output_dir="./out",
        orpo_data_path="data.jsonl",
    )
    with pytest.raises(AttributeError):
        opts.lambda_orpo = 2.0  # type: ignore[misc]


def test_orpo_example_creation() -> None:
    """OrpoExample stores preference pair."""
    ex = OrpoExample(prompt="hi", chosen="good", rejected="bad")
    assert ex.prompt == "hi"
    assert ex.chosen == "good"
    assert ex.rejected == "bad"
