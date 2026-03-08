"""Unit tests for training run comparison logic."""

from __future__ import annotations

from serve.model_comparison import (
    RunComparisonReport,
    compare_training_runs,
    format_comparison_report,
    _compute_config_delta,
)
from serve.training_run_types import TrainingRunRecord


def _make_run_record(
    run_id: str = "run-001",
    dataset_name: str = "demo",
    output_dir: str = "/tmp/out",
    config_hash: str = "abc123",
    state: str = "completed",
    created_at: str = "2026-01-01T00:00:00Z",
) -> TrainingRunRecord:
    """Build a test TrainingRunRecord with sensible defaults."""
    return TrainingRunRecord(
        run_id=run_id,
        dataset_name=dataset_name,
        output_dir=output_dir,
        parent_model_path=None,
        config_hash=config_hash,
        state=state,
        created_at=created_at,
        updated_at=created_at,
        events=(),
    )


def test_compare_identical_runs() -> None:
    """Identical config fields should produce an empty delta."""
    run_a = _make_run_record(run_id="run-a")
    run_b = _make_run_record(run_id="run-b")
    report = compare_training_runs(run_a, run_b)

    assert report.config_delta == {}
    assert report.run_id_a == "run-a"
    assert report.run_id_b == "run-b"


def test_compare_different_configs() -> None:
    """Different config hashes should appear in the delta."""
    run_a = _make_run_record(run_id="run-a", config_hash="hash-aaa")
    run_b = _make_run_record(run_id="run-b", config_hash="hash-bbb")
    report = compare_training_runs(run_a, run_b)

    assert "config_hash" in report.config_delta
    assert report.config_delta["config_hash"] == ("hash-aaa", "hash-bbb")


def test_compare_different_datasets() -> None:
    """Different dataset names should be reflected in the report."""
    run_a = _make_run_record(run_id="run-a", dataset_name="alpha")
    run_b = _make_run_record(run_id="run-b", dataset_name="beta")
    report = compare_training_runs(run_a, run_b)

    assert report.dataset_a == "alpha"
    assert report.dataset_b == "beta"
    assert "dataset_name" in report.config_delta
    assert report.config_delta["dataset_name"] == ("alpha", "beta")


def test_format_report_includes_run_ids() -> None:
    """Formatted report lines should include both run IDs."""
    run_a = _make_run_record(run_id="run-aaa")
    run_b = _make_run_record(run_id="run-bbb")
    report = compare_training_runs(run_a, run_b)
    lines = format_comparison_report(report)
    joined = "\n".join(lines)

    assert "run-aaa" in joined
    assert "run-bbb" in joined


def test_config_delta_only_differing_keys() -> None:
    """Matching keys must be excluded from the delta dict."""
    config_a = {"lr": 0.01, "epochs": 10, "batch_size": 32}
    config_b = {"lr": 0.001, "epochs": 10, "batch_size": 32}
    delta = _compute_config_delta(config_a, config_b)

    assert "lr" in delta
    assert delta["lr"] == (0.01, 0.001)
    assert "epochs" not in delta
    assert "batch_size" not in delta


def test_format_report_shows_no_diff_message() -> None:
    """Report with no config delta should show a 'none' message."""
    run_a = _make_run_record(run_id="run-a")
    run_b = _make_run_record(run_id="run-b")
    report = compare_training_runs(run_a, run_b)
    lines = format_comparison_report(report)
    joined = "\n".join(lines)

    assert "Config differences: none" in joined
