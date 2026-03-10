"""Unit tests for benchmark runner orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.benchmark_types import (
    BenchmarkConfig,
    BenchmarkResult,
    LatencyResult,
    PerplexityResult,
)
from core.types import DataRecord, RecordMetadata
from serve.benchmark_runner import format_benchmark_report, run_benchmark


def _build_records() -> list[DataRecord]:
    """Build sample records for benchmark tests."""
    metadata = RecordMetadata(
        source_uri="a.txt",
        language="en",
        quality_score=0.9,
        perplexity=1.5,
    )
    return [
        DataRecord(record_id="id-1", text="alpha beta gamma delta", metadata=metadata),
        DataRecord(record_id="id-2", text="epsilon zeta eta theta", metadata=metadata),
    ]


def _build_perplexity_result() -> PerplexityResult:
    return PerplexityResult(perplexity=42.5, num_tokens=100, num_sequences=2)


def _build_latency_result() -> LatencyResult:
    return LatencyResult(
        mean_latency_ms=5.0,
        p50_latency_ms=4.5,
        p95_latency_ms=8.0,
        p99_latency_ms=10.0,
        throughput_tokens_per_sec=200.0,
    )


def test_run_benchmark_skips_perplexity_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Runner should skip perplexity when run_perplexity is False."""
    config = BenchmarkConfig(
        model_path=str(tmp_path / "model.pt"),
        dataset_name="demo",
        output_dir=str(tmp_path / "out"),
        run_perplexity=False,
        run_latency=True,
    )
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    monkeypatch.setattr(
        "serve.benchmark_runner._import_torch",
        lambda: SimpleNamespace(
            no_grad=MagicMock(return_value=MagicMock(
                __enter__=MagicMock(), __exit__=MagicMock(return_value=False),
            )),
            tensor=lambda d, dtype=None: MagicMock(
                to=MagicMock(return_value=MagicMock(size=MagicMock(return_value=4))),
                size=MagicMock(return_value=4),
            ),
            long=0,
        ),
    )
    monkeypatch.setattr(
        "serve.benchmark_runner._load_model",
        lambda *a, **kw: (mock_model, "cpu"),
    )
    monkeypatch.setattr(
        "serve.benchmark_runner._tokenize_records",
        lambda r, m: [[1, 2, 3]],
    )
    monkeypatch.setattr(
        "serve.benchmark_runner.profile_latency",
        lambda **kw: _build_latency_result(),
    )

    result = run_benchmark(_build_records(), config, tmp_path)

    assert result.perplexity is None
    assert result.latency is not None


def test_run_benchmark_skips_latency_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Runner should skip latency when run_latency is False."""
    config = BenchmarkConfig(
        model_path=str(tmp_path / "model.pt"),
        dataset_name="demo",
        output_dir=str(tmp_path / "out"),
        run_perplexity=True,
        run_latency=False,
    )
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    monkeypatch.setattr(
        "serve.benchmark_runner._import_torch",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "serve.benchmark_runner._load_model",
        lambda *a, **kw: (mock_model, "cpu"),
    )
    monkeypatch.setattr(
        "serve.benchmark_runner._tokenize_records",
        lambda r, m: [[1, 2, 3]],
    )
    monkeypatch.setattr(
        "serve.benchmark_runner.compute_perplexity_benchmark",
        lambda **kw: _build_perplexity_result(),
    )

    result = run_benchmark(_build_records(), config, tmp_path)

    assert result.perplexity is not None
    assert result.latency is None


def test_run_benchmark_runs_both_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Runner should run both benchmarks when both flags are True."""
    config = BenchmarkConfig(
        model_path=str(tmp_path / "model.pt"),
        dataset_name="demo",
        output_dir=str(tmp_path / "out"),
        run_perplexity=True,
        run_latency=True,
    )
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    monkeypatch.setattr(
        "serve.benchmark_runner._import_torch",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "serve.benchmark_runner._load_model",
        lambda *a, **kw: (mock_model, "cpu"),
    )
    monkeypatch.setattr(
        "serve.benchmark_runner._tokenize_records",
        lambda r, m: [[1, 2, 3]],
    )
    monkeypatch.setattr(
        "serve.benchmark_runner.compute_perplexity_benchmark",
        lambda **kw: _build_perplexity_result(),
    )
    monkeypatch.setattr(
        "serve.benchmark_runner.profile_latency",
        lambda **kw: _build_latency_result(),
    )

    result = run_benchmark(_build_records(), config, tmp_path)

    assert result.perplexity is not None
    assert result.latency is not None


def test_run_benchmark_saves_report_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Runner should save text and JSON report files."""
    out_dir = tmp_path / "out"
    config = BenchmarkConfig(
        model_path=str(tmp_path / "model.pt"),
        dataset_name="demo",
        output_dir=str(out_dir),
        run_perplexity=False,
        run_latency=False,
    )
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    monkeypatch.setattr(
        "serve.benchmark_runner._import_torch",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "serve.benchmark_runner._load_model",
        lambda *a, **kw: (mock_model, "cpu"),
    )
    monkeypatch.setattr(
        "serve.benchmark_runner._tokenize_records",
        lambda r, m: [[1, 2, 3]],
    )

    run_benchmark(_build_records(), config, tmp_path)

    assert (out_dir / "benchmark_report.txt").exists()
    assert (out_dir / "benchmark_report.json").exists()


def test_format_benchmark_report_includes_perplexity() -> None:
    """Report formatter should include perplexity section."""
    result = BenchmarkResult(
        model_path="/tmp/model.pt",
        dataset_name="demo",
        perplexity=_build_perplexity_result(),
        latency=None,
    )

    report = format_benchmark_report(result)

    assert "Perplexity" in report
    assert "42.5" in report
    assert "Latency" not in report


def test_format_benchmark_report_includes_latency() -> None:
    """Report formatter should include latency section."""
    result = BenchmarkResult(
        model_path="/tmp/model.pt",
        dataset_name="demo",
        perplexity=None,
        latency=_build_latency_result(),
    )

    report = format_benchmark_report(result)

    assert "Latency" in report
    assert "5.000" in report
    assert "Perplexity" not in report


def test_format_benchmark_report_both_sections() -> None:
    """Report formatter should include both sections when present."""
    result = BenchmarkResult(
        model_path="/tmp/model.pt",
        dataset_name="demo",
        perplexity=_build_perplexity_result(),
        latency=_build_latency_result(),
    )

    report = format_benchmark_report(result)

    assert "Perplexity" in report
    assert "Latency" in report
    assert "Crucible Benchmark Report" in report
