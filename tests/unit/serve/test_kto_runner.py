"""Tests for KTO training runner."""

from __future__ import annotations

import pytest

from core.kto_types import KtoExample, KtoOptions


def test_kto_options_defaults() -> None:
    """KtoOptions defaults are set correctly."""
    opts = KtoOptions(
        dataset_name="test", output_dir="./out",
        kto_data_path="data.jsonl",
    )
    assert opts.beta == 0.1
    assert opts.desirable_weight == 1.0
    assert opts.undesirable_weight == 1.0
    assert opts.epochs == 3


def test_kto_options_custom() -> None:
    """KtoOptions accepts custom weights."""
    opts = KtoOptions(
        dataset_name="test", output_dir="./out",
        kto_data_path="data.jsonl",
        beta=0.2, desirable_weight=1.5, undesirable_weight=0.5,
    )
    assert opts.beta == 0.2
    assert opts.desirable_weight == 1.5
    assert opts.undesirable_weight == 0.5


def test_kto_options_frozen() -> None:
    """KtoOptions is immutable."""
    opts = KtoOptions(
        dataset_name="test", output_dir="./out",
        kto_data_path="data.jsonl",
    )
    with pytest.raises(AttributeError):
        opts.beta = 0.5  # type: ignore[misc]


def test_kto_example_creation() -> None:
    """KtoExample stores binary label correctly."""
    desirable = KtoExample(prompt="hi", response="hello", is_desirable=True)
    undesirable = KtoExample(prompt="hi", response="bad", is_desirable=False)
    assert desirable.is_desirable is True
    assert undesirable.is_desirable is False
