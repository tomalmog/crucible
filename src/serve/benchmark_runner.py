"""Benchmark runner orchestrating perplexity and latency evaluation.

This module loads a trained model, optionally runs perplexity and
latency benchmarks, and returns a combined result with a text report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.benchmark_types import (
    BenchmarkConfig,
    BenchmarkResult,
    LatencyResult,
    PerplexityResult,
)
from core.errors import ForgeBenchmarkError, ForgeDependencyError
from core.types import DataRecord
from serve.benchmark_latency import profile_latency
from serve.benchmark_perplexity import compute_perplexity_benchmark
from serve.device_selection import resolve_execution_device
from serve.tokenization import VocabularyTokenizer


def run_benchmark(
    records: list[DataRecord],
    config: BenchmarkConfig,
    data_root: Path,
) -> BenchmarkResult:
    """Run benchmark evaluation on a trained model.

    Args:
        records: Dataset records for evaluation.
        config: Benchmark configuration.
        data_root: Root data directory.

    Returns:
        Combined benchmark result.

    Raises:
        ForgeBenchmarkError: If model loading or evaluation fails.
        ForgeDependencyError: If torch is unavailable.
    """
    torch_module = _import_torch()
    model, device = _load_model(torch_module, config.model_path)
    sequences = _tokenize_records(records, config.max_token_length)
    perplexity = _run_perplexity(
        torch_module, model, sequences, device, config,
    )
    latency = _run_latency(
        torch_module, model, sequences, device, config,
    )
    result = BenchmarkResult(
        model_path=config.model_path,
        dataset_name=config.dataset_name,
        perplexity=perplexity,
        latency=latency,
    )
    _save_report(result, config.output_dir)
    return result


def format_benchmark_report(result: BenchmarkResult) -> str:
    """Format a human-readable benchmark summary.

    Args:
        result: Combined benchmark result.

    Returns:
        Multi-line text report string.
    """
    lines = [
        "=== Forge Benchmark Report ===",
        f"Model: {result.model_path}",
        f"Dataset: {result.dataset_name}",
    ]
    if result.perplexity is not None:
        lines.append("")
        lines.append("--- Perplexity ---")
        lines.append(f"  Perplexity: {result.perplexity.perplexity:.4f}")
        lines.append(f"  Tokens: {result.perplexity.num_tokens}")
        lines.append(f"  Sequences: {result.perplexity.num_sequences}")
    if result.latency is not None:
        lines.append("")
        lines.append("--- Latency ---")
        lines.append(
            f"  Mean: {result.latency.mean_latency_ms:.3f} ms"
        )
        lines.append(
            f"  P50: {result.latency.p50_latency_ms:.3f} ms"
        )
        lines.append(
            f"  P95: {result.latency.p95_latency_ms:.3f} ms"
        )
        lines.append(
            f"  P99: {result.latency.p99_latency_ms:.3f} ms"
        )
        lines.append(
            f"  Throughput: "
            f"{result.latency.throughput_tokens_per_sec:.2f} tok/s"
        )
    return "\n".join(lines)


def _import_torch() -> Any:
    """Import torch dependency."""
    try:
        import torch
    except ImportError as error:
        raise ForgeDependencyError(
            "Benchmark requires torch. Install torch to run benchmarks."
        ) from error
    return torch


def _load_model(
    torch_module: Any, model_path: str,
) -> tuple[Any, Any]:
    """Load model weights and return model and device."""
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise ForgeBenchmarkError(
            f"Model file not found at {path}."
        )
    device = resolve_execution_device(torch_module)
    try:
        model = torch_module.load(str(path), map_location=device)
    except Exception as error:
        raise ForgeBenchmarkError(
            f"Failed to load model from {path}: {error}"
        ) from error
    model.eval()
    return model, device


def _tokenize_records(
    records: list[DataRecord], max_token_length: int,
) -> list[list[int]]:
    """Tokenize dataset records into integer sequences."""
    tokenizer = VocabularyTokenizer.create()
    tokenizer.fit(record.text for record in records)
    sequences = []
    for record in records:
        encoded = tokenizer.encode(record.text, max_token_length)
        if len(encoded) > 1:
            sequences.append(encoded)
    if not sequences:
        raise ForgeBenchmarkError(
            "No valid sequences generated from records."
        )
    return sequences


def _run_perplexity(
    torch_module: Any,
    model: Any,
    sequences: list[list[int]],
    device: Any,
    config: BenchmarkConfig,
) -> PerplexityResult | None:
    """Run perplexity benchmark if enabled."""
    if not config.run_perplexity:
        return None
    return compute_perplexity_benchmark(
        torch_module=torch_module,
        model=model,
        sequences=sequences,
        device=device,
        batch_size=config.batch_size,
    )


def _run_latency(
    torch_module: Any,
    model: Any,
    sequences: list[list[int]],
    device: Any,
    config: BenchmarkConfig,
) -> LatencyResult | None:
    """Run latency profiling if enabled."""
    if not config.run_latency:
        return None
    return profile_latency(
        torch_module=torch_module,
        model=model,
        sequences=sequences,
        device=device,
    )


def _save_report(result: BenchmarkResult, output_dir: str) -> None:
    """Save benchmark report as text and JSON to output dir."""
    out_path = Path(output_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    report_text = format_benchmark_report(result)
    (out_path / "benchmark_report.txt").write_text(
        report_text, encoding="utf-8",
    )
    report_data = _result_to_dict(result)
    (out_path / "benchmark_report.json").write_text(
        json.dumps(report_data, indent=2), encoding="utf-8",
    )


def _result_to_dict(result: BenchmarkResult) -> dict[str, object]:
    """Convert benchmark result to a serializable dictionary."""
    data: dict[str, object] = {
        "model_path": result.model_path,
        "dataset_name": result.dataset_name,
    }
    if result.perplexity is not None:
        data["perplexity"] = {
            "perplexity": result.perplexity.perplexity,
            "num_tokens": result.perplexity.num_tokens,
            "num_sequences": result.perplexity.num_sequences,
        }
    if result.latency is not None:
        data["latency"] = {
            "mean_latency_ms": result.latency.mean_latency_ms,
            "p50_latency_ms": result.latency.p50_latency_ms,
            "p95_latency_ms": result.latency.p95_latency_ms,
            "p99_latency_ms": result.latency.p99_latency_ms,
            "throughput_tokens_per_sec": (
                result.latency.throughput_tokens_per_sec
            ),
        }
    return data
