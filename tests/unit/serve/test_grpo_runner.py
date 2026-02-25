"""Tests for GRPO training runner."""

from __future__ import annotations

import pytest

from core.grpo_types import GrpoOptions


def test_grpo_options_defaults() -> None:
    """GrpoOptions defaults are set correctly."""
    opts = GrpoOptions(
        dataset_name="test",
        output_dir="./out",
        grpo_data_path="data.jsonl",
    )
    assert opts.group_size == 4
    assert opts.kl_coeff == 0.1
    assert opts.clip_range == 0.2
    assert opts.temperature == 1.0
    assert opts.epochs == 3
    assert opts.batch_size == 16


def test_grpo_options_custom() -> None:
    """GrpoOptions accepts custom values."""
    opts = GrpoOptions(
        dataset_name="test",
        output_dir="./out",
        grpo_data_path="data.jsonl",
        group_size=8,
        kl_coeff=0.05,
        clip_range=0.3,
        temperature=0.7,
    )
    assert opts.group_size == 8
    assert opts.kl_coeff == 0.05
    assert opts.clip_range == 0.3
    assert opts.temperature == 0.7


def test_grpo_options_frozen() -> None:
    """GrpoOptions is immutable."""
    opts = GrpoOptions(
        dataset_name="test",
        output_dir="./out",
        grpo_data_path="data.jsonl",
    )
    with pytest.raises(AttributeError):
        opts.group_size = 16  # type: ignore[misc]
