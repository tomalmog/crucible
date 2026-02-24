"""Unit tests for GPU telemetry collection."""

from __future__ import annotations

import time
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from serve.training_telemetry import GpuTelemetryCollector, TelemetrySnapshot


class _FakeCudaModule:
    """Fake cuda module that returns predictable memory and utilization."""

    def __init__(
        self,
        allocated_bytes: float = 1024.0 * 1024.0 * 512.0,
        reserved_bytes: float = 1024.0 * 1024.0 * 1024.0,
        utilization_value: float = 75.0,
    ) -> None:
        self._allocated_bytes = allocated_bytes
        self._reserved_bytes = reserved_bytes
        self._utilization_value = utilization_value

    def memory_allocated(self) -> float:
        return self._allocated_bytes

    def memory_reserved(self) -> float:
        return self._reserved_bytes

    def utilization(self) -> float:
        return self._utilization_value


class _FakeTorchModule:
    """Fake torch module with a cuda sub-module."""

    def __init__(self, cuda: Any = None) -> None:
        self.cuda = cuda


class _FakeDevice:
    """Fake torch device with a type attribute."""

    def __init__(self, device_type: str) -> None:
        self.type = device_type


def test_telemetry_snapshot_on_cpu_returns_zeros() -> None:
    """On a CPU device, all GPU metrics should be zero."""
    torch_module = _FakeTorchModule(cuda=None)
    device = _FakeDevice("cpu")
    collector = GpuTelemetryCollector(torch_module, device)

    snapshot = collector.snapshot()

    assert snapshot.gpu_memory_allocated_mb == 0.0
    assert snapshot.gpu_memory_reserved_mb == 0.0
    assert snapshot.gpu_utilization_percent is None
    assert snapshot.tokens_per_second == 0.0


def test_record_batch_increments_token_count() -> None:
    """Recording batches should accumulate tokens and produce throughput."""
    torch_module = _FakeTorchModule(cuda=None)
    device = _FakeDevice("cpu")
    collector = GpuTelemetryCollector(torch_module, device)

    collector.record_batch(100)
    collector.record_batch(200)

    # Allow a tiny amount of time to pass so throughput can be calculated
    time.sleep(0.01)
    snapshot = collector.snapshot()

    assert snapshot.tokens_per_second > 0.0


def test_reset_clears_accumulated_state() -> None:
    """After reset, token count and timing window should be cleared."""
    torch_module = _FakeTorchModule(cuda=None)
    device = _FakeDevice("cpu")
    collector = GpuTelemetryCollector(torch_module, device)

    collector.record_batch(500)
    time.sleep(0.01)
    collector.reset()

    snapshot = collector.snapshot()

    assert snapshot.tokens_per_second == 0.0


def test_telemetry_snapshot_frozen() -> None:
    """TelemetrySnapshot should be immutable."""
    snapshot = TelemetrySnapshot(
        gpu_memory_allocated_mb=100.0,
        gpu_memory_reserved_mb=200.0,
        gpu_utilization_percent=50.0,
        tokens_per_second=1000.0,
        timestamp_seconds=0.0,
    )

    with pytest.raises(FrozenInstanceError):
        snapshot.gpu_memory_allocated_mb = 999.0  # type: ignore[misc]
