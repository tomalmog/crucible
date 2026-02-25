"""End-to-end integration tests for the evaluation harness workflow.

Tests cover the EvaluationHarness SDK methods (evaluate, list, load)
and CLI eval subcommand using real file I/O against a temporary directory.
"""

from __future__ import annotations

from pathlib import Path

from cli.main import main
from eval.benchmark_runner import AVAILABLE_BENCHMARKS, BenchmarkResult, EvaluationResult
from eval.evaluation_harness import EvaluationHarness


def test_all_benchmarks(tmp_path: Path) -> None:
    """Evaluating with no benchmark filter should return all 7 benchmarks."""
    harness = EvaluationHarness(tmp_path)

    result = harness.evaluate("model.pt")

    assert isinstance(result, EvaluationResult)
    assert len(result.benchmark_results) == 7
    assert result.average_score >= 0
    names = {r.benchmark_name for r in result.benchmark_results}
    assert names == set(AVAILABLE_BENCHMARKS)
    for br in result.benchmark_results:
        assert isinstance(br, BenchmarkResult)
        assert br.num_examples > 0


def test_selected_benchmarks(tmp_path: Path) -> None:
    """Evaluating with a benchmark filter should return exactly those benchmarks."""
    harness = EvaluationHarness(tmp_path)

    result = harness.evaluate("model.pt", benchmarks=["mmlu", "gsm8k"])

    assert len(result.benchmark_results) == 2
    names = {r.benchmark_name for r in result.benchmark_results}
    assert names == {"mmlu", "gsm8k"}


def test_store_and_load(tmp_path: Path) -> None:
    """Evaluating should store results that can be listed and loaded."""
    harness = EvaluationHarness(tmp_path)
    harness.evaluate("model.pt")

    eval_ids = harness.list_evaluations()

    assert len(eval_ids) >= 1
    loaded = harness.load_evaluation(eval_ids[0])
    assert "model_path" in loaded
    assert loaded["model_path"] == "model.pt"
    assert "benchmarks" in loaded
    assert len(loaded["benchmarks"]) == 7


def test_base_model_comparison(tmp_path: Path) -> None:
    """Evaluating with a base model should produce non-empty base_results."""
    harness = EvaluationHarness(tmp_path)

    result = harness.evaluate(
        "model.pt",
        base_model_path="base.pt",
    )

    assert result.base_model_path == "base.pt"
    assert isinstance(result.base_results, tuple)
    assert len(result.base_results) > 0
    assert len(result.base_results) == len(result.benchmark_results)


def test_cli_selected(tmp_path: Path, capsys) -> None:
    """CLI eval with --benchmarks filter should exit 0 and print selected names."""
    exit_code = main([
        "--data-root", str(tmp_path),
        "eval",
        "--model-path", "m.pt",
        "--benchmarks", "mmlu,gsm8k",
    ])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "mmlu" in captured
    assert "gsm8k" in captured


def test_cli_all(tmp_path: Path, capsys) -> None:
    """CLI eval without --benchmarks should exit 0 and print all 7 benchmark names."""
    exit_code = main([
        "--data-root", str(tmp_path),
        "eval",
        "--model-path", "m.pt",
    ])
    captured = capsys.readouterr().out

    assert exit_code == 0
    for name in AVAILABLE_BENCHMARKS:
        assert name in captured
