"""Latency profiling benchmark for trained models.

This module times forward passes over tokenized sequences and
computes latency percentiles and throughput metrics.
"""

from __future__ import annotations

import time
from typing import Any

from core.benchmark_types import LatencyResult
from core.errors import ForgeBenchmarkError


def profile_latency(
    torch_module: Any,
    model: Any,
    sequences: list[list[int]],
    device: Any,
    num_runs: int = 50,
) -> LatencyResult:
    """Profile inference latency over tokenized sequences.

    Args:
        torch_module: Imported torch module.
        model: PyTorch model in eval mode.
        sequences: List of token-id sequences.
        device: Torch device for computation.
        num_runs: Number of forward passes to measure.

    Returns:
        LatencyResult with percentile and throughput data.

    Raises:
        ForgeBenchmarkError: If no sequences or profiling fails.
    """
    if not sequences:
        raise ForgeBenchmarkError(
            "No sequences provided for latency profiling."
        )
    model.eval()
    sample_sequence = sequences[0]
    input_tensor = _prepare_input(
        torch_module, sample_sequence, device,
    )
    token_count = input_tensor.size(1)
    latencies = _measure_forward_passes(
        torch_module, model, input_tensor, num_runs,
    )
    return _compute_latency_stats(latencies, token_count)


def _prepare_input(
    torch_module: Any, sequence: list[int], device: Any,
) -> Any:
    """Build a batched input tensor from a single sequence."""
    tensor = torch_module.tensor([sequence], dtype=torch_module.long)
    return tensor.to(device)


def _measure_forward_passes(
    torch_module: Any,
    model: Any,
    input_tensor: Any,
    num_runs: int,
) -> list[float]:
    """Time individual forward passes and return latencies in ms."""
    latencies: list[float] = []
    with torch_module.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)
    return latencies


def _compute_latency_stats(
    latencies: list[float], token_count: int,
) -> LatencyResult:
    """Compute percentile statistics from latency measurements."""
    if not latencies:
        raise ForgeBenchmarkError(
            "No latency measurements recorded."
        )
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    mean_ms = sum(sorted_latencies) / n
    p50_ms = _percentile(sorted_latencies, 0.50)
    p95_ms = _percentile(sorted_latencies, 0.95)
    p99_ms = _percentile(sorted_latencies, 0.99)
    mean_seconds = mean_ms / 1000.0
    throughput = (
        token_count / mean_seconds if mean_seconds > 0 else 0.0
    )
    return LatencyResult(
        mean_latency_ms=round(mean_ms, 3),
        p50_latency_ms=round(p50_ms, 3),
        p95_latency_ms=round(p95_ms, 3),
        p99_latency_ms=round(p99_ms, 3),
        throughput_tokens_per_sec=round(throughput, 2),
    )


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute a percentile from pre-sorted values."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = pct * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] + fraction * (
        sorted_values[upper] - sorted_values[lower]
    )
