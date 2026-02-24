"""Unit tests for the FSDP parallelism strategy."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

from serve.fsdp_strategy import FsdpConfig, FsdpStrategy


def test_fsdp_config_frozen() -> None:
    """FsdpConfig should be a frozen dataclass."""
    config = FsdpConfig()

    with pytest.raises(FrozenInstanceError):
        config.sharding_strategy = "no_shard"  # type: ignore[misc]


def test_strategy_name_is_fsdp() -> None:
    """FsdpStrategy.strategy_name() should return 'fsdp'."""
    strategy = FsdpStrategy()

    assert strategy.strategy_name() == "fsdp"


def test_fsdp_config_defaults() -> None:
    """FsdpConfig should have sensible default values."""
    config = FsdpConfig()

    assert config.sharding_strategy == "full_shard"
    assert config.auto_wrap_min_params == 100_000
    assert config.cpu_offload is False


def test_wrap_model_calls_fsdp() -> None:
    """wrap_model should call FullyShardedDataParallel on the model."""
    mock_torch = MagicMock()
    mock_fsdp_class = MagicMock()
    mock_fsdp_instance = MagicMock()
    mock_fsdp_class.return_value = mock_fsdp_instance
    mock_torch.distributed.fsdp.FullyShardedDataParallel = mock_fsdp_class
    mock_torch.distributed.fsdp.ShardingStrategy.FULL_SHARD = "FULL_SHARD"
    mock_torch.distributed.fsdp.CPUOffload = MagicMock()
    mock_torch.distributed.fsdp.wrap.size_based_auto_wrap_policy = None
    mock_torch.distributed.is_initialized.return_value = True
    mock_torch.distributed.get_rank.return_value = 0

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_device = MagicMock()

    strategy = FsdpStrategy(FsdpConfig(sharding_strategy="full_shard"))
    with patch("serve.fsdp_strategy._import_torch", return_value=mock_torch):
        result = strategy.wrap_model(mock_model, mock_device)

    mock_fsdp_class.assert_called_once()
    call_kwargs = mock_fsdp_class.call_args
    assert call_kwargs[0][0] is mock_model
    assert call_kwargs[1]["sharding_strategy"] == "FULL_SHARD"
    assert result is mock_fsdp_instance
