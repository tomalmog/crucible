"""Unit tests for the DDP training runner structural setup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.errors import ForgeDistributedError
from serve.ddp_training_runner import (
    _resolve_rank_device,
    _unwrap_ddp_model,
    _wrap_model_with_ddp,
)


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    def __init__(self, cuda_available: bool = False) -> None:
        self.cuda = _FakeCuda(cuda_available)
        self.nn = _FakeNN()

    def device(self, value: str) -> str:
        return value


class _FakeNN:
    def __init__(self) -> None:
        self.parallel = _FakeParallel()


class _FakeParallel:
    def __init__(self) -> None:
        self.DistributedDataParallel = _FakeDDP


class _FakeDDP:
    """Mock DDP wrapper that stores the original module."""

    def __init__(self, model: object, device_ids: list[int] | None = None) -> None:
        self.module = model
        self.device_ids = device_ids


def test_resolve_rank_device_returns_cuda_for_rank() -> None:
    """Should return cuda:<rank> when CUDA is available."""
    torch_mod = _FakeTorch(cuda_available=True)

    device = _resolve_rank_device(torch_mod, rank=2)

    assert device == "cuda:2"


def test_resolve_rank_device_falls_back_to_cpu() -> None:
    """Should return cpu when CUDA is unavailable."""
    torch_mod = _FakeTorch(cuda_available=False)

    device = _resolve_rank_device(torch_mod, rank=0)

    assert device == "cpu"


def test_wrap_model_with_ddp_returns_wrapped_model() -> None:
    """wrap_model_with_ddp should return a DDP-wrapped model."""
    torch_mod = _FakeTorch()
    model = MagicMock()

    wrapped = _wrap_model_with_ddp(torch_mod, model, rank=1)

    assert isinstance(wrapped, _FakeDDP)
    assert wrapped.module is model
    assert wrapped.device_ids == [1]


def test_wrap_model_raises_when_parallel_unavailable() -> None:
    """Should raise ForgeDistributedError when nn.parallel is missing."""
    torch_mod = _FakeTorch()
    torch_mod.nn.parallel = None  # type: ignore[assignment]
    model = MagicMock()

    with pytest.raises(ForgeDistributedError, match="unavailable"):
        _wrap_model_with_ddp(torch_mod, model, rank=0)


def test_wrap_model_raises_when_ddp_class_unavailable() -> None:
    """Should raise ForgeDistributedError when DDP class is missing."""
    torch_mod = _FakeTorch()
    torch_mod.nn.parallel.DistributedDataParallel = None  # type: ignore[assignment]
    model = MagicMock()

    with pytest.raises(ForgeDistributedError, match="not available"):
        _wrap_model_with_ddp(torch_mod, model, rank=0)


def test_unwrap_ddp_model_returns_inner_module() -> None:
    """unwrap_ddp_model should return the .module attribute."""
    inner_model = MagicMock()
    wrapped = _FakeDDP(inner_model)

    result = _unwrap_ddp_model(wrapped)

    assert result is inner_model


def test_unwrap_ddp_model_returns_model_when_no_module() -> None:
    """unwrap_ddp_model should return the model itself if no .module."""
    plain_model = MagicMock(spec=[])  # no .module attribute

    result = _unwrap_ddp_model(plain_model)

    assert result is plain_model


def test_save_rank_zero_artifacts_skips_on_non_zero_rank() -> None:
    """Rank != 0 should produce empty artifact paths."""
    from serve.ddp_training_runner import _save_rank_zero_artifacts

    torch_mod = MagicMock()
    model = MagicMock()
    metrics = [{"epoch": 1.0, "train_loss": 0.5, "validation_loss": 0.6}]

    result = _save_rank_zero_artifacts(
        torch_module=torch_mod,
        model=model,
        epoch_metrics=metrics,
        output_dir=MagicMock(),
        rank=1,
    )

    assert result.model_path == ""
    assert result.history_path == ""
    assert result.epochs_completed == 1
