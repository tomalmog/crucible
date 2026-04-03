"""Tests for benchmark_runner resilience: per-benchmark exception isolation,
base model comparison error handling, partial result writing, and edge cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from eval.benchmark_runner import (
    AVAILABLE_BENCHMARKS,
    BenchmarkResult,
    EvaluationResult,
    run_benchmarks,
)


def _make_eval_model() -> MagicMock:
    model = MagicMock()
    model.torch_module = MagicMock()
    model.torch_module.cuda = MagicMock()
    model.torch_module.cuda.is_available.return_value = False
    return model


def _make_passing_result(name: str, score: float = 75.0) -> BenchmarkResult:
    return BenchmarkResult(benchmark_name=name, score=score, num_examples=10, correct=7)


# ── Per-benchmark isolation ──────────────────────────────────────────────────


def test_failing_benchmark_zero_score_recorded(tmp_path: Path) -> None:
    """A single benchmark failure records score=0 with error detail; others run normally."""
    with (
        patch("eval.benchmarks._model_loader.load_eval_model", return_value=_make_eval_model()),
        patch("eval.benchmarks.mmlu.run_mmlu", side_effect=RuntimeError("OOM")),
        patch("eval.benchmarks.gsm8k.run_gsm8k", return_value=_make_passing_result("gsm8k")),
    ):
        result = run_benchmarks("model.pt", ["mmlu", "gsm8k"])

    assert len(result.benchmark_results) == 2
    mmlu = next(r for r in result.benchmark_results if r.benchmark_name == "mmlu")
    gsm8k = next(r for r in result.benchmark_results if r.benchmark_name == "gsm8k")
    assert mmlu.score == 0.0
    assert "error" in mmlu.details
    assert "OOM" in mmlu.details["error"]
    assert gsm8k.score == 75.0


def test_one_failing_benchmark_does_not_abort_others() -> None:
    """A failing benchmark must NOT stop remaining benchmarks from executing."""
    ran = []

    def run_mmlu(model_path, *, max_samples=None, eval_model=None):
        ran.append("mmlu")
        raise RuntimeError("mmlu dead")

    def run_gsm8k(model_path, *, max_samples=None, eval_model=None):
        ran.append("gsm8k")
        return _make_passing_result("gsm8k")

    def run_arc(model_path, *, max_samples=None, eval_model=None):
        ran.append("arc")
        return _make_passing_result("arc")

    with (
        patch("eval.benchmarks._model_loader.load_eval_model", return_value=_make_eval_model()),
        patch("eval.benchmarks.mmlu.run_mmlu", side_effect=run_mmlu),
        patch("eval.benchmarks.gsm8k.run_gsm8k", side_effect=run_gsm8k),
        patch("eval.benchmarks.arc.run_arc", side_effect=run_arc),
    ):
        result = run_benchmarks("model.pt", ["mmlu", "gsm8k", "arc"])

    assert "mmlu" in ran
    assert "gsm8k" in ran
    assert "arc" in ran
    assert len(result.benchmark_results) == 3


def test_all_benchmarks_fail_returns_zero_average() -> None:
    """If all benchmarks fail, average score is 0.0 and all results are recorded."""
    with (
        patch("eval.benchmarks._model_loader.load_eval_model", return_value=_make_eval_model()),
        patch("eval.benchmarks.mmlu.run_mmlu", side_effect=ValueError("bad")),
        patch("eval.benchmarks.gsm8k.run_gsm8k", side_effect=ValueError("bad")),
    ):
        result = run_benchmarks("model.pt", ["mmlu", "gsm8k"])

    assert result.average_score == 0.0
    assert len(result.benchmark_results) == 2
    assert all(r.score == 0.0 for r in result.benchmark_results)


# ── Base model comparison error isolation ────────────────────────────────────


def test_base_model_benchmark_failure_records_zero_score() -> None:
    """A base model benchmark failure records 0 score and continues remaining benchmarks."""
    call_count = {"mmlu": 0, "gsm8k": 0}

    def run_mmlu(model_path, *, max_samples=None, eval_model=None):
        call_count["mmlu"] += 1
        return _make_passing_result("mmlu")

    def run_gsm8k(model_path, *, max_samples=None, eval_model=None):
        call_count["gsm8k"] += 1
        if call_count["gsm8k"] > 1:
            raise RuntimeError("base model gsm8k dead")
        return _make_passing_result("gsm8k")

    with (
        patch("eval.benchmarks._model_loader.load_eval_model", return_value=_make_eval_model()),
        patch("eval.benchmarks.mmlu.run_mmlu", side_effect=run_mmlu),
        patch("eval.benchmarks.gsm8k.run_gsm8k", side_effect=run_gsm8k),
    ):
        result = run_benchmarks("model.pt", ["mmlu", "gsm8k"], base_model_path="base.pt")

    assert len(result.benchmark_results) == 2
    assert result.benchmark_results[0].score == 75.0
    assert len(result.base_results) == 2
    failed_base = next(r for r in result.base_results if r.benchmark_name == "gsm8k")
    assert failed_base.score == 0.0
    assert "error" in failed_base.details


def test_base_model_all_fail_fine_model_unaffected() -> None:
    """All base benchmarks failing doesn't affect fine model scores."""
    call_count = [0]

    def run_mmlu(model_path, *, max_samples=None, eval_model=None):
        call_count[0] += 1
        if call_count[0] > 1:
            raise RuntimeError("base dead")
        return _make_passing_result("mmlu", score=90.0)

    with (
        patch("eval.benchmarks._model_loader.load_eval_model", return_value=_make_eval_model()),
        patch("eval.benchmarks.mmlu.run_mmlu", side_effect=run_mmlu),
    ):
        result = run_benchmarks("model.pt", ["mmlu"], base_model_path="base.pt")

    assert result.benchmark_results[0].score == 90.0
    assert len(result.base_results) == 1
    assert result.base_results[0].score == 0.0


def test_base_model_benchmarks_all_run_even_if_first_fails() -> None:
    """When base model is provided and first benchmark fails, remaining still run."""
    ran = []
    call_count = {"mmlu": 0, "arc": 0}

    def run_mmlu(model_path, *, max_samples=None, eval_model=None):
        call_count["mmlu"] += 1
        if call_count["mmlu"] > 1:  # second call is for base model
            ran.append("base-mmlu")
            raise RuntimeError("base mmlu dead")
        ran.append("mmlu")
        return _make_passing_result("mmlu")

    def run_arc(model_path, *, max_samples=None, eval_model=None):
        call_count["arc"] += 1
        if call_count["arc"] > 1:
            ran.append("base-arc")
        else:
            ran.append("arc")
        return _make_passing_result("arc")

    with (
        patch("eval.benchmarks._model_loader.load_eval_model", return_value=_make_eval_model()),
        patch("eval.benchmarks.mmlu.run_mmlu", side_effect=run_mmlu),
        patch("eval.benchmarks.arc.run_arc", side_effect=run_arc),
    ):
        result = run_benchmarks("model.pt", ["mmlu", "arc"], base_model_path="base.pt")

    assert "base-arc" in ran, "base model arc benchmark must run even if base mmlu fails"
    assert len(result.base_results) == 2


# ── Partial result writing ────────────────────────────────────────────────────


def test_partial_results_written_after_each_benchmark(tmp_path: Path) -> None:
    """_write_partial_results is called once per completed benchmark."""
    write_calls = []

    def capture_write(path, model_path, results, total):
        write_calls.append(len(results))

    with (
        patch("eval.benchmarks._model_loader.load_eval_model", return_value=_make_eval_model()),
        patch("eval.benchmarks.mmlu.run_mmlu", return_value=_make_passing_result("mmlu")),
        patch("eval.benchmarks.gsm8k.run_gsm8k", return_value=_make_passing_result("gsm8k")),
        patch("eval.benchmark_runner._write_partial_results", wraps=capture_write),
    ):
        run_benchmarks("model.pt", ["mmlu", "gsm8k"], output_path=str(tmp_path / "r.json"))

    assert write_calls == [1, 2]  # called after each benchmark with cumulative count


def test_partial_result_status_transitions() -> None:
    """First write is 'partial'; last write is 'completed'."""
    statuses = []

    def capture_write(path, model_path, results, total):
        statuses.append("completed" if len(results) >= total else "partial")

    with (
        patch("eval.benchmarks._model_loader.load_eval_model", return_value=_make_eval_model()),
        patch("eval.benchmarks.mmlu.run_mmlu", return_value=_make_passing_result("mmlu")),
        patch("eval.benchmarks.gsm8k.run_gsm8k", return_value=_make_passing_result("gsm8k")),
        patch("eval.benchmark_runner._write_partial_results", wraps=capture_write),
    ):
        run_benchmarks("model.pt", ["mmlu", "gsm8k"], output_path="r.json")

    assert statuses[0] == "partial"
    assert statuses[-1] == "completed"


# ── Input validation ──────────────────────────────────────────────────────────


def test_unknown_benchmark_raises_clearly() -> None:
    from core.errors import CrucibleBenchmarkError
    with pytest.raises(CrucibleBenchmarkError, match="No valid benchmark names"):
        run_benchmarks("model.pt", ["definitely_not_real"])


def test_empty_benchmarks_list_raises() -> None:
    from core.errors import CrucibleBenchmarkError
    with pytest.raises(CrucibleBenchmarkError):
        run_benchmarks("model.pt", [])


def test_mixed_valid_and_invalid_runs_only_valid() -> None:
    """Valid benchmarks run; invalid names are skipped without crashing."""
    with (
        patch("eval.benchmarks._model_loader.load_eval_model", return_value=_make_eval_model()),
        patch("eval.benchmarks.mmlu.run_mmlu", return_value=_make_passing_result("mmlu")),
    ):
        result = run_benchmarks("model.pt", ["mmlu", "nonexistent_benchmark_xyz"])

    assert len(result.benchmark_results) == 1
    assert result.benchmark_results[0].benchmark_name == "mmlu"


# ── Average score correctness ─────────────────────────────────────────────────


def test_average_score_rounded_to_two_decimals() -> None:
    with (
        patch("eval.benchmarks._model_loader.load_eval_model", return_value=_make_eval_model()),
        patch("eval.benchmarks.mmlu.run_mmlu", return_value=BenchmarkResult(
            benchmark_name="mmlu", score=66.666666, num_examples=3, correct=2,
        )),
        patch("eval.benchmarks.gsm8k.run_gsm8k", return_value=BenchmarkResult(
            benchmark_name="gsm8k", score=33.333333, num_examples=3, correct=1,
        )),
    ):
        result = run_benchmarks("model.pt", ["mmlu", "gsm8k"])

    assert result.average_score == 50.0


def test_single_benchmark_average_equals_its_score() -> None:
    with (
        patch("eval.benchmarks._model_loader.load_eval_model", return_value=_make_eval_model()),
        patch("eval.benchmarks.mmlu.run_mmlu", return_value=_make_passing_result("mmlu", score=83.5)),
    ):
        result = run_benchmarks("model.pt", ["mmlu"])

    assert result.average_score == 83.5


def test_single_failed_benchmark_average_is_zero() -> None:
    with (
        patch("eval.benchmarks._model_loader.load_eval_model", return_value=_make_eval_model()),
        patch("eval.benchmarks.mmlu.run_mmlu", side_effect=RuntimeError("dead")),
    ):
        result = run_benchmarks("model.pt", ["mmlu"])

    assert result.average_score == 0.0


# ── EvaluationResult structure ────────────────────────────────────────────────


def test_result_contains_model_path() -> None:
    with (
        patch("eval.benchmarks._model_loader.load_eval_model", return_value=_make_eval_model()),
        patch("eval.benchmarks.mmlu.run_mmlu", return_value=_make_passing_result("mmlu")),
    ):
        result = run_benchmarks("/path/to/model.pt", ["mmlu"])

    assert result.model_path == "/path/to/model.pt"


def test_result_base_model_path_preserved() -> None:
    with (
        patch("eval.benchmarks._model_loader.load_eval_model", return_value=_make_eval_model()),
        patch("eval.benchmarks.mmlu.run_mmlu", return_value=_make_passing_result("mmlu")),
    ):
        result = run_benchmarks("model.pt", ["mmlu"], base_model_path="/base/model.pt")

    assert result.base_model_path == "/base/model.pt"
