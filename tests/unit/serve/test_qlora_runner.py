"""Tests for QLoRA training runner."""

from __future__ import annotations

import pytest

from core.qlora_types import QloraOptions


def test_qlora_options_defaults() -> None:
    """QloraOptions defaults are set correctly."""
    opts = QloraOptions(
        dataset_name="test", output_dir="./out",
        qlora_data_path="data.jsonl", base_model_path="model.pt",
    )
    assert opts.quantization_bits == 4
    assert opts.qlora_type == "nf4"
    assert opts.double_quantize is True
    assert opts.lora_rank == 8
    assert opts.lora_alpha == 16.0
    assert opts.lora_dropout == 0.0


def test_qlora_options_custom_bits() -> None:
    """QloraOptions accepts 8-bit quantization."""
    opts = QloraOptions(
        dataset_name="test", output_dir="./out",
        qlora_data_path="data.jsonl", base_model_path="model.pt",
        quantization_bits=8,
    )
    assert opts.quantization_bits == 8


def test_qlora_options_frozen() -> None:
    """QloraOptions is immutable."""
    opts = QloraOptions(
        dataset_name="test", output_dir="./out",
        qlora_data_path="data.jsonl", base_model_path="model.pt",
    )
    with pytest.raises(AttributeError):
        opts.lora_rank = 16  # type: ignore[misc]
