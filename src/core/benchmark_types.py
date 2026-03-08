"""Typed benchmark evaluation models.

This module defines immutable data models for benchmark configuration
and results used by the benchmark runner and CLI command.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a benchmark evaluation run.

    Attributes:
        model_path: Path to trained model weights.
        dataset_name: Dataset to evaluate against.
        output_dir: Directory for benchmark report output.
        max_token_length: Maximum token sequence length.
        batch_size: Batch size for evaluation passes.
        run_perplexity: Whether to compute perplexity metric.
        run_latency: Whether to profile inference latency.
    """

    model_path: str
    dataset_name: str
    output_dir: str
    max_token_length: int = 512
    batch_size: int = 16
    run_perplexity: bool = True
    run_latency: bool = True


@dataclass(frozen=True)
class PerplexityResult:
    """Perplexity evaluation result.

    Attributes:
        perplexity: Exponential of average cross-entropy loss.
        num_tokens: Total tokens evaluated.
        num_sequences: Total sequences evaluated.
    """

    perplexity: float
    num_tokens: int
    num_sequences: int


@dataclass(frozen=True)
class LatencyResult:
    """Latency profiling result.

    Attributes:
        mean_latency_ms: Mean forward-pass latency in milliseconds.
        p50_latency_ms: Median latency in milliseconds.
        p95_latency_ms: 95th percentile latency in milliseconds.
        p99_latency_ms: 99th percentile latency in milliseconds.
        throughput_tokens_per_sec: Token throughput rate.
    """

    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_tokens_per_sec: float


@dataclass(frozen=True)
class BenchmarkResult:
    """Combined benchmark evaluation result.

    Attributes:
        model_path: Path to the evaluated model.
        dataset_name: Dataset used for evaluation.
        perplexity: Perplexity result if computed.
        latency: Latency result if profiled.
    """

    model_path: str
    dataset_name: str
    perplexity: PerplexityResult | None = None
    latency: LatencyResult | None = None
