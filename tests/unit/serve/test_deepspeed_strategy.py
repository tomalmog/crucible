"""Unit tests for the DeepSpeed parallelism strategy."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from serve.deepspeed_strategy import (
    DeepSpeedConfig,
    DeepSpeedStrategy,
    build_deepspeed_config_dict,
)


def test_deepspeed_config_frozen() -> None:
    """DeepSpeedConfig should be a frozen dataclass."""
    config = DeepSpeedConfig()

    with pytest.raises(FrozenInstanceError):
        config.zero_stage = 3  # type: ignore[misc]


def test_strategy_name_is_deepspeed() -> None:
    """DeepSpeedStrategy.strategy_name() should return 'deepspeed'."""
    strategy = DeepSpeedStrategy()

    assert strategy.strategy_name() == "deepspeed"


def test_build_config_dict_structure() -> None:
    """build_deepspeed_config_dict should produce required keys."""
    config = DeepSpeedConfig(zero_stage=2)

    result = build_deepspeed_config_dict(config, lr=1e-4, batch_size=8)

    assert "train_micro_batch_size_per_gpu" in result
    assert result["train_micro_batch_size_per_gpu"] == 8
    assert "gradient_clipping" in result
    assert result["gradient_clipping"] == 1.0
    assert "zero_optimization" in result
    assert "optimizer" in result
    assert result["optimizer"]["type"] == "AdamW"
    assert result["optimizer"]["params"]["lr"] == 1e-4


def test_build_config_dict_zero_stage() -> None:
    """build_deepspeed_config_dict should set the correct ZeRO stage."""
    for stage in (0, 1, 2, 3):
        config = DeepSpeedConfig(zero_stage=stage)
        result = build_deepspeed_config_dict(config, lr=1e-3, batch_size=4)

        assert result["zero_optimization"]["stage"] == stage
