"""GPU telemetry collection for training monitoring.

This module samples GPU memory, utilization, and throughput metrics during
training runs, providing structured telemetry data for progress reporting.

Assumptions:
- torch is passed as a module reference (not imported directly).
- On non-CUDA devices, zero-valued snapshots are returned gracefully.
- Older PyTorch versions missing cuda.utilization() are handled safely.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TelemetrySnapshot:
    """One telemetry sample taken during training.

    Attributes:
        gpu_memory_allocated_mb: Currently allocated GPU memory in MB.
        gpu_memory_reserved_mb: Reserved (cached) GPU memory in MB.
        gpu_utilization_percent: GPU core utilization percentage, or None.
        tokens_per_second: Throughput measured over the collection window.
        timestamp_seconds: Monotonic timestamp when snapshot was taken.
    """

    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float
    gpu_utilization_percent: float | None
    tokens_per_second: float
    timestamp_seconds: float


_BYTES_PER_MB: float = 1024.0 * 1024.0


class GpuTelemetryCollector:
    """Collects GPU telemetry samples during training.

    Call record_batch() after each batch to track throughput.
    Call snapshot() to read current GPU state.
    """

    def __init__(self, torch_module: Any, device: Any) -> None:
        self._torch_module = torch_module
        self._device = device
        self._total_tokens: int = 0
        self._window_start: float = time.monotonic()

    def record_batch(self, token_count: int) -> None:
        """Record tokens processed in one batch for throughput calculation.

        Side-effects:
            Increments the internal token accumulator.
        """
        self._total_tokens += token_count

    def snapshot(self) -> TelemetrySnapshot:
        """Take a current GPU telemetry reading.

        Returns:
            Telemetry snapshot with memory, utilization, and throughput.
        """
        now = time.monotonic()
        tokens_per_second = _compute_throughput(
            self._total_tokens, self._window_start, now,
        )
        if not _is_cuda_device(self._device):
            return _zero_snapshot(tokens_per_second, now)
        memory_allocated = _read_memory_allocated(self._torch_module)
        memory_reserved = _read_memory_reserved(self._torch_module)
        utilization = _read_gpu_utilization(self._torch_module)
        return TelemetrySnapshot(
            gpu_memory_allocated_mb=memory_allocated,
            gpu_memory_reserved_mb=memory_reserved,
            gpu_utilization_percent=utilization,
            tokens_per_second=tokens_per_second,
            timestamp_seconds=now,
        )

    def reset(self) -> None:
        """Reset accumulated counters for a new measurement window."""
        self._total_tokens = 0
        self._window_start = time.monotonic()


def _is_cuda_device(device: Any) -> bool:
    """Check whether the device is a CUDA device."""
    device_type = getattr(device, "type", None)
    if isinstance(device_type, str):
        return device_type == "cuda"
    return str(device) == "cuda"


def _compute_throughput(
    total_tokens: int,
    window_start: float,
    now: float,
) -> float:
    """Compute tokens-per-second from accumulated counters."""
    elapsed = now - window_start
    if elapsed <= 0.0 or total_tokens <= 0:
        return 0.0
    return total_tokens / elapsed


def _read_memory_allocated(torch_module: Any) -> float:
    """Read current GPU memory allocation in MB."""
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is None:
        return 0.0
    memory_fn = getattr(cuda_module, "memory_allocated", None)
    if not callable(memory_fn):
        return 0.0
    return float(memory_fn()) / _BYTES_PER_MB


def _read_memory_reserved(torch_module: Any) -> float:
    """Read current GPU memory reservation in MB."""
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is None:
        return 0.0
    memory_fn = getattr(cuda_module, "memory_reserved", None)
    if not callable(memory_fn):
        return 0.0
    return float(memory_fn()) / _BYTES_PER_MB


def _read_gpu_utilization(torch_module: Any) -> float | None:
    """Read GPU utilization percentage, returning None if unavailable."""
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is None:
        return None
    util_fn = getattr(cuda_module, "utilization", None)
    if not callable(util_fn):
        return None
    try:
        return float(util_fn())
    except (RuntimeError, OSError):
        return None


def _zero_snapshot(tokens_per_second: float, now: float) -> TelemetrySnapshot:
    """Build a zero-valued snapshot for non-CUDA devices."""
    return TelemetrySnapshot(
        gpu_memory_allocated_mb=0.0,
        gpu_memory_reserved_mb=0.0,
        gpu_utilization_percent=None,
        tokens_per_second=tokens_per_second,
        timestamp_seconds=now,
    )
