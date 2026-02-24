"""Unit tests for LoRA adapter I/O operations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.errors import ForgeLoraError
from core.lora_types import LoraAdapterInfo, LoraConfig


def test_lora_adapter_info_frozen() -> None:
    """LoraAdapterInfo should be immutable."""
    info = LoraAdapterInfo(
        adapter_path="/tmp/adapter.pt",
        rank=8,
        alpha=16.0,
        target_modules=("q_proj", "v_proj"),
    )
    with pytest.raises(AttributeError):
        info.rank = 16  # type: ignore[misc]


def test_lora_adapter_info_optional_base_model() -> None:
    """LoraAdapterInfo should default base_model_path to None."""
    info = LoraAdapterInfo(
        adapter_path="/tmp/adapter.pt",
        rank=8,
        alpha=16.0,
        target_modules=("q_proj",),
    )
    assert info.base_model_path is None


def test_lora_config_defaults() -> None:
    """LoraConfig should have expected default values."""
    config = LoraConfig()
    assert config.rank == 8
    assert config.alpha == 16.0
    assert config.dropout == 0.0
    assert "q_proj" in config.target_modules
