"""Integration tests for cost tracking end-to-end workflow.

Covers CostTracker SDK methods and CLI cost subcommands
using real file I/O against a temporary directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cli.main import main
from serve.cost_tracker import CostTracker, RunCost, ProjectCostSummary


def test_log_and_retrieve(tmp_path: Path) -> None:
    """Logging a 1-hour run should produce correct gpu_hours and electricity."""
    tracker = CostTracker(tmp_path)

    cost = tracker.log_run_cost(
        run_id="run-1",
        duration_seconds=3600.0,
        gpu_type="A100",
        tdp_watts=300,
        electricity_rate_per_kwh=0.12,
    )

    assert isinstance(cost, RunCost)
    assert cost.gpu_hours == pytest.approx(1.0, abs=1e-4)
    assert cost.gpu_type == "A100"
    # electricity_kwh = (300 / 1000) * 1.0 = 0.3
    assert cost.electricity_kwh == pytest.approx(0.3, abs=1e-4)
    # electricity_cost = 0.3 * 0.12 = 0.036
    assert cost.electricity_cost_usd == pytest.approx(0.036, abs=1e-4)
    assert cost.total_cost_usd == pytest.approx(0.036, abs=1e-4)

    # Verify persistence via get_run_cost
    loaded = tracker.get_run_cost("run-1")
    assert loaded is not None
    assert loaded.run_id == "run-1"
    assert loaded.gpu_hours == cost.gpu_hours


def test_missing_run(tmp_path: Path) -> None:
    """Querying a nonexistent run should return None."""
    tracker = CostTracker(tmp_path)

    result = tracker.get_run_cost("nonexistent")

    assert result is None


def test_project_summary(tmp_path: Path) -> None:
    """Summary should aggregate totals across all logged runs."""
    tracker = CostTracker(tmp_path)
    tracker.log_run_cost("r1", duration_seconds=3600.0, tdp_watts=300)
    tracker.log_run_cost("r2", duration_seconds=7200.0, tdp_watts=300)
    tracker.log_run_cost("r3", duration_seconds=1800.0, tdp_watts=300)

    summary = tracker.get_project_summary()

    assert isinstance(summary, ProjectCostSummary)
    assert summary.total_runs == 3
    # gpu_hours: 1.0 + 2.0 + 0.5 = 3.5
    assert summary.total_gpu_hours == pytest.approx(3.5, abs=1e-4)
    # electricity_kwh: each at 300W => (0.3) * hours
    # 0.3 + 0.6 + 0.15 = 1.05
    assert summary.total_electricity_kwh == pytest.approx(1.05, abs=1e-4)
    # total_cost = sum of individual electricity costs (no cloud cost)
    # 0.3*0.12 + 0.6*0.12 + 0.15*0.12 = 0.036 + 0.072 + 0.018 = 0.126
    assert summary.total_cost_usd == pytest.approx(0.126, abs=1e-4)
    assert len(summary.runs) == 3


def test_empty_summary(tmp_path: Path) -> None:
    """Summary with no runs should show zero totals."""
    tracker = CostTracker(tmp_path)

    summary = tracker.get_project_summary()

    assert summary.total_runs == 0
    assert summary.total_gpu_hours == 0.0
    assert summary.total_electricity_kwh == 0.0
    assert summary.total_cost_usd == 0.0


def test_cli_summary(tmp_path: Path, capsys) -> None:
    """CLI 'cost summary' should print totals to stdout."""
    tracker = CostTracker(tmp_path)
    tracker.log_run_cost("cli-r1", duration_seconds=3600.0, gpu_type="V100")

    exit_code = main(["--data-root", str(tmp_path), "cost", "summary"])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "total_runs=1" in captured
    assert "total_gpu_hours=" in captured
    assert "cli-r1" in captured


def test_cli_run_detail(tmp_path: Path, capsys) -> None:
    """CLI 'cost run' should print gpu_hours for a specific run."""
    tracker = CostTracker(tmp_path)
    tracker.log_run_cost("detail-r1", duration_seconds=7200.0, gpu_type="A100")

    exit_code = main(["--data-root", str(tmp_path), "cost", "run", "detail-r1"])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "gpu_hours=" in captured
    assert "detail-r1" in captured
