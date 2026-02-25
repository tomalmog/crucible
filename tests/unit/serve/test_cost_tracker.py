"""Tests for cost tracker."""

from __future__ import annotations

from pathlib import Path

from serve.cost_tracker import CostTracker, RunCost


def test_log_and_retrieve_cost(tmp_path: Path) -> None:
    """Log a run cost and retrieve it."""
    tracker = CostTracker(tmp_path)
    cost = tracker.log_run_cost(
        run_id="run-1", duration_seconds=3600,
        gpu_type="rtx4090", tdp_watts=450,
    )
    assert cost.gpu_hours == 1.0
    assert cost.electricity_kwh > 0
    retrieved = tracker.get_run_cost("run-1")
    assert retrieved is not None
    assert retrieved.run_id == "run-1"


def test_project_summary(tmp_path: Path) -> None:
    """Aggregate project summary."""
    tracker = CostTracker(tmp_path)
    tracker.log_run_cost("run-1", 3600, "rtx4090", 450)
    tracker.log_run_cost("run-2", 7200, "a100", 400)
    summary = tracker.get_project_summary()
    assert summary.total_runs == 2
    assert summary.total_gpu_hours == 3.0


def test_empty_summary(tmp_path: Path) -> None:
    """Empty project has zero costs."""
    tracker = CostTracker(tmp_path)
    summary = tracker.get_project_summary()
    assert summary.total_runs == 0
    assert summary.total_cost_usd == 0.0


def test_nonexistent_run(tmp_path: Path) -> None:
    """Nonexistent run returns None."""
    tracker = CostTracker(tmp_path)
    assert tracker.get_run_cost("nonexistent") is None


def test_cost_with_cloud(tmp_path: Path) -> None:
    """Cost includes cloud compute charges."""
    tracker = CostTracker(tmp_path)
    cost = tracker.log_run_cost(
        "run-1", 3600, cloud_cost_usd=5.0,
    )
    assert cost.total_cost_usd >= 5.0
