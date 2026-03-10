"""Tests for TPU parallelism strategy."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.errors import CrucibleDistributedError


def _make_mock_torch() -> MagicMock:
    """Build a mock torch module."""
    torch_mock = MagicMock()
    torch_mock.optim.AdamW.return_value = MagicMock(name="adamw_optimizer")
    return torch_mock


def _make_mock_xla(device: Any = "xla:0", world_size: int = 4) -> MagicMock:
    """Build a mock torch_xla module."""
    xla = MagicMock()
    xla.xla_device.return_value = device
    xla.xrt_world_size.return_value = world_size
    xla.runtime = SimpleNamespace(initialize_cache=MagicMock())
    xla.mark_step = MagicMock()
    xla.save = MagicMock()
    return xla


def _make_mock_model() -> MagicMock:
    """Build a mock PyTorch model."""
    model = MagicMock()
    param1 = MagicMock()
    param1.requires_grad = True
    param2 = MagicMock()
    param2.requires_grad = False
    model.parameters.return_value = [param1, param2]
    model.to.return_value = model
    model.state_dict.return_value = {"weight": "data"}
    return model


class TestTpuStrategy:
    """Tests for TpuStrategy class."""

    def _build_strategy(
        self, torch_mock: MagicMock | None = None, xla_mock: MagicMock | None = None,
    ) -> Any:
        """Build a TpuStrategy with mocked dependencies."""
        torch_mod = torch_mock or _make_mock_torch()
        xla_mod = xla_mock or _make_mock_xla()
        with patch("serve.tpu_strategy._import_torch", return_value=torch_mod), \
             patch("serve.tpu_strategy.import_xla", return_value=xla_mod):
            from serve.tpu_strategy import TpuStrategy
            return TpuStrategy()

    def test_strategy_name(self) -> None:
        strategy = self._build_strategy()
        assert strategy.strategy_name() == "tpu"

    def test_wrap_model_moves_to_xla_device(self) -> None:
        model = _make_mock_model()
        xla = _make_mock_xla(device="xla:0")
        strategy = self._build_strategy(xla_mock=xla)
        with patch("serve.tpu_strategy.resolve_tpu_device", return_value="xla:0"), \
             patch("serve.tpu_strategy.init_xla_mesh"):
            result = strategy.wrap_model(model, "cpu")
        model.to.assert_called_once_with("xla:0")
        assert result is model

    def test_build_optimizer_filters_trainable_params(self) -> None:
        torch_mock = _make_mock_torch()
        strategy = self._build_strategy(torch_mock=torch_mock)
        model = _make_mock_model()
        optimizer = strategy.build_optimizer(model, lr=1e-4, weight_decay=0.01)
        call_args = torch_mock.optim.AdamW.call_args
        params = call_args[0][0]
        assert len(params) == 1
        assert params[0].requires_grad is True

    def test_save_checkpoint_uses_xla_save(self) -> None:
        xla = _make_mock_xla()
        strategy = self._build_strategy(xla_mock=xla)
        model = _make_mock_model()
        optimizer = MagicMock()
        optimizer.state_dict.return_value = {"lr": 1e-4}
        with patch("serve.tpu_strategy.resolve_tpu_device", return_value="xla:0"):
            strategy.save_checkpoint(model, optimizer, "/tmp/ckpt.pt")
        xla.mark_step.assert_called_once()
        xla.save.assert_called_once()
        saved_state = xla.save.call_args[0][0]
        assert "model_state_dict" in saved_state
        assert "optimizer_state_dict" in saved_state

    def test_save_checkpoint_raises_on_error(self) -> None:
        xla = _make_mock_xla()
        xla.mark_step.side_effect = RuntimeError("sync fail")
        strategy = self._build_strategy(xla_mock=xla)
        model = _make_mock_model()
        optimizer = MagicMock()
        with pytest.raises(CrucibleDistributedError, match="Failed to save TPU checkpoint"):
            strategy.save_checkpoint(model, optimizer, "/tmp/ckpt.pt")

    def test_load_checkpoint_loads_state(self) -> None:
        torch_mock = _make_mock_torch()
        xla = _make_mock_xla(device="xla:0")
        strategy = self._build_strategy(torch_mock=torch_mock, xla_mock=xla)
        model = _make_mock_model()
        optimizer = MagicMock()
        torch_mock.load.return_value = {
            "model_state_dict": {"weight": "loaded"},
            "optimizer_state_dict": {"lr": 1e-3},
        }
        with patch("serve.tpu_strategy.resolve_tpu_device", return_value="xla:0"):
            strategy.load_checkpoint(model, optimizer, "/tmp/ckpt.pt")
        model.load_state_dict.assert_called_once_with({"weight": "loaded"})
        optimizer.load_state_dict.assert_called_once_with({"lr": 1e-3})

    def test_load_checkpoint_raises_on_error(self) -> None:
        torch_mock = _make_mock_torch()
        xla = _make_mock_xla(device="xla:0")
        strategy = self._build_strategy(torch_mock=torch_mock, xla_mock=xla)
        torch_mock.load.side_effect = FileNotFoundError("not found")
        model = _make_mock_model()
        optimizer = MagicMock()
        with patch("serve.tpu_strategy.resolve_tpu_device", return_value="xla:0"):
            with pytest.raises(CrucibleDistributedError, match="Failed to load TPU checkpoint"):
                strategy.load_checkpoint(model, optimizer, "/tmp/ckpt.pt")
