"""Model latency profiling for deployment readiness.

This module measures forward-pass latency across batch size and
sequence length combinations, producing percentile statistics.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from core.deployment_types import LatencyProfile
from core.errors import ForgeDeployError, ForgeDependencyError


def profile_model_latency(
    model_path: str,
    batch_sizes: tuple[int, ...],
    sequence_lengths: tuple[int, ...],
    device: str = "cpu",
    num_runs: int = 10,
) -> tuple[LatencyProfile, ...]:
    """Profile inference latency across batch and sequence configurations.

    Args:
        model_path: Path to the ONNX model file.
        batch_sizes: Batch sizes to profile.
        sequence_lengths: Sequence lengths to profile.
        device: Device string for inference.
        num_runs: Number of forward passes per configuration.

    Returns:
        Tuple of LatencyProfile results for each combination.

    Raises:
        ForgeDeployError: If model path is invalid.
        ForgeDependencyError: If onnxruntime is not available.
    """
    if not os.path.isfile(model_path):
        raise ForgeDeployError(f"Model file not found: {model_path}")

    session = _load_onnx_session(model_path)
    profiles: list[LatencyProfile] = []

    for batch_size in batch_sizes:
        for seq_len in sequence_lengths:
            profile = _profile_single_config(
                session, batch_size, seq_len, device, num_runs,
            )
            profiles.append(profile)

    return tuple(profiles)


def _load_onnx_session(model_path: str) -> Any:
    """Load an ONNX inference session.

    Args:
        model_path: Path to the ONNX model.

    Returns:
        An onnxruntime InferenceSession.

    Raises:
        ForgeDependencyError: If onnxruntime is not installed.
    """
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ForgeDependencyError(
            "onnxruntime is required for latency profiling. "
            "Install with: pip install onnxruntime"
        ) from exc

    return ort.InferenceSession(model_path)


def _profile_single_config(
    model: Any,
    batch_size: int,
    seq_len: int,
    device: str,
    num_runs: int,
) -> LatencyProfile:
    """Time forward passes for a single batch/sequence configuration.

    Args:
        model: ONNX inference session.
        batch_size: Number of samples per batch.
        seq_len: Token count per sequence.
        device: Compute device identifier.
        num_runs: Number of timed forward passes.

    Returns:
        LatencyProfile with percentile statistics.
    """
    input_name = model.get_inputs()[0].name
    dummy_input = np.random.randn(batch_size, seq_len).astype(
        np.float32,
    )

    latencies: list[float] = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model.run(None, {input_name: dummy_input})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)

    arr = np.array(latencies)
    total_tokens = batch_size * seq_len
    mean_ms = float(np.mean(arr))
    throughput = (
        total_tokens / (mean_ms / 1000.0) if mean_ms > 0 else 0.0
    )

    return LatencyProfile(
        batch_size=batch_size,
        sequence_length=seq_len,
        device=device,
        mean_latency_ms=mean_ms,
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        throughput_tokens_per_sec=throughput,
    )


def format_latency_report(
    profiles: tuple[LatencyProfile, ...],
) -> tuple[str, ...]:
    """Format latency profiles as human-readable table lines.

    Args:
        profiles: Tuple of profiling results.

    Returns:
        Tuple of formatted report lines.
    """
    header = (
        f"{'Batch':>6} {'SeqLen':>7} {'Mean(ms)':>9} "
        f"{'P50(ms)':>8} {'P95(ms)':>8} {'P99(ms)':>8} "
        f"{'Tok/s':>10}"
    )
    lines: list[str] = [header, "-" * len(header)]

    for p in profiles:
        line = (
            f"{p.batch_size:>6} {p.sequence_length:>7} "
            f"{p.mean_latency_ms:>9.2f} {p.p50_ms:>8.2f} "
            f"{p.p95_ms:>8.2f} {p.p99_ms:>8.2f} "
            f"{p.throughput_tokens_per_sec:>10.1f}"
        )
        lines.append(line)

    return tuple(lines)
