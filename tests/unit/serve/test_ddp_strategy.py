"""Unit tests for the DDP parallelism strategy."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from serve.ddp_strategy import DdpStrategy


def _build_mock_torch(rank: int = 0) -> MagicMock:
    """Build a mock torch module with DDP support."""
    mock_torch = MagicMock()
    mock_torch.distributed.is_initialized.return_value = True
    mock_torch.distributed.get_rank.return_value = rank
    return mock_torch


def test_strategy_name_is_ddp() -> None:
    """DdpStrategy.strategy_name() should return 'ddp'."""
    strategy = DdpStrategy()

    assert strategy.strategy_name() == "ddp"


def test_wrap_model_calls_ddp() -> None:
    """wrap_model should call DistributedDataParallel on the model."""
    mock_torch = _build_mock_torch()
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_device = MagicMock()
    mock_device.index = 0
    mock_ddp_instance = MagicMock()
    mock_torch.nn.parallel.DistributedDataParallel.return_value = (
        mock_ddp_instance
    )

    strategy = DdpStrategy()
    with patch("serve.ddp_strategy._import_torch", return_value=mock_torch):
        result = strategy.wrap_model(mock_model, mock_device)

    mock_torch.nn.parallel.DistributedDataParallel.assert_called_once_with(
        mock_model, device_ids=[0],
    )
    assert result is mock_ddp_instance


def test_save_checkpoint_rank0_only() -> None:
    """save_checkpoint should only save on rank 0."""
    mock_torch_rank0 = _build_mock_torch(rank=0)
    mock_torch_rank1 = _build_mock_torch(rank=1)
    mock_model = MagicMock()
    mock_model.module = MagicMock()
    mock_optimizer = MagicMock()
    strategy = DdpStrategy()

    with patch(
        "serve.ddp_strategy._import_torch",
        return_value=mock_torch_rank0,
    ), patch(
        "serve.distributed_setup._get_dist_module",
        return_value=mock_torch_rank0.distributed,
    ):
        strategy.save_checkpoint(mock_model, mock_optimizer, "/tmp/ckpt.pt")
    mock_torch_rank0.save.assert_called_once()

    mock_torch_rank1.save.reset_mock()
    with patch(
        "serve.ddp_strategy._import_torch",
        return_value=mock_torch_rank1,
    ), patch(
        "serve.distributed_setup._get_dist_module",
        return_value=mock_torch_rank1.distributed,
    ):
        strategy.save_checkpoint(mock_model, mock_optimizer, "/tmp/ckpt.pt")
    mock_torch_rank1.save.assert_not_called()


def test_build_optimizer() -> None:
    """build_optimizer should return an AdamW optimizer."""
    mock_torch = _build_mock_torch()
    mock_model = MagicMock()
    mock_optimizer = MagicMock()
    mock_torch.optim.AdamW.return_value = mock_optimizer

    strategy = DdpStrategy()
    with patch("serve.ddp_strategy._import_torch", return_value=mock_torch):
        result = strategy.build_optimizer(mock_model, lr=1e-3, weight_decay=0.01)

    mock_torch.optim.AdamW.assert_called_once_with(
        mock_model.parameters(), lr=1e-3, weight_decay=0.01,
    )
    assert result is mock_optimizer
