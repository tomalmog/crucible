"""Integration tests for experiment tracking end-to-end workflow.

Covers ExperimentTracker SDK methods and CLI experiment subcommands
using real file I/O against a temporary directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from cli.main import main
from serve.experiment_tracker import ExperimentTracker, MetricEntry


def test_log_and_retrieve_metrics(tmp_path: Path) -> None:
    """Logging metrics should produce retrievable MetricEntry objects."""
    tracker = ExperimentTracker(tmp_path)
    tracker.log_metrics("run-1", step=1, metrics={"loss": 0.5})

    entries = tracker.get_run_metrics("run-1")

    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, MetricEntry)
    assert entry.step == 1
    assert entry.name == "loss"
    assert entry.value == 0.5
    assert entry.timestamp > 0


def test_log_hyperparameters_and_hardware(tmp_path: Path) -> None:
    """Hyperparameters and hardware info should appear in the run summary."""
    tracker = ExperimentTracker(tmp_path)
    tracker.log_metrics("run-hp", step=1, metrics={"loss": 0.4})
    tracker.log_hyperparameters("run-hp", {"lr": 1e-4, "epochs": 3})
    tracker.log_hardware("run-hp", {"gpu": "A100", "vram_gb": 80})

    summary = tracker.get_run_summary("run-hp")

    assert summary["hyperparameters"]["lr"] == 1e-4
    assert summary["hyperparameters"]["epochs"] == 3
    assert summary["hardware"]["gpu"] == "A100"
    assert summary["hardware"]["vram_gb"] == 80


def test_run_summary_aggregates(tmp_path: Path) -> None:
    """Summary should compute final, min, and max for each metric."""
    tracker = ExperimentTracker(tmp_path)
    tracker.log_metrics("run-agg", step=1, metrics={"loss": 0.9})
    tracker.log_metrics("run-agg", step=2, metrics={"loss": 0.5})
    tracker.log_metrics("run-agg", step=3, metrics={"loss": 0.3})

    summary = tracker.get_run_summary("run-agg")

    assert summary["loss_final"] == 0.3
    assert summary["loss_min"] == 0.3
    assert summary["loss_max"] == 0.9
    assert "loss" in summary["metric_names"]


def test_list_runs_returns_all(tmp_path: Path) -> None:
    """All created runs should appear in list_runs output."""
    tracker = ExperimentTracker(tmp_path)
    run_ids = ["alpha", "bravo", "charlie"]
    for rid in run_ids:
        tracker.log_metrics(rid, step=1, metrics={"loss": 0.1})

    listed = tracker.list_runs()

    assert len(listed) == 3
    for rid in run_ids:
        assert rid in listed


def test_compare_runs(tmp_path: Path) -> None:
    """Comparing two runs should return summaries for both."""
    tracker = ExperimentTracker(tmp_path)
    tracker.log_metrics("run-a", step=1, metrics={"loss": 0.8})
    tracker.log_metrics("run-b", step=1, metrics={"loss": 0.4})

    comparisons = tracker.compare_runs(["run-a", "run-b"])

    assert len(comparisons) == 2
    ids = [c["run_id"] for c in comparisons]
    assert "run-a" in ids
    assert "run-b" in ids
    a_summary = next(c for c in comparisons if c["run_id"] == "run-a")
    b_summary = next(c for c in comparisons if c["run_id"] == "run-b")
    assert a_summary["loss_final"] == 0.8
    assert b_summary["loss_final"] == 0.4


def test_delete_run(tmp_path: Path) -> None:
    """Deleting a run should remove it from the list."""
    tracker = ExperimentTracker(tmp_path)
    tracker.log_metrics("run-del", step=1, metrics={"loss": 0.6})
    assert "run-del" in tracker.list_runs()

    deleted = tracker.delete_run_metrics("run-del")

    assert deleted is True
    assert "run-del" not in tracker.list_runs()


def test_cli_list(tmp_path: Path, capsys) -> None:
    """CLI 'experiment list' should print run IDs to stdout."""
    tracker = ExperimentTracker(tmp_path)
    tracker.log_metrics("cli-run-1", step=1, metrics={"loss": 0.5})

    exit_code = main(["--data-root", str(tmp_path), "experiment", "list"])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "cli-run-1" in captured


def test_cli_show_and_compare(tmp_path: Path, capsys) -> None:
    """CLI 'experiment show' and 'experiment compare' should output JSON."""
    tracker = ExperimentTracker(tmp_path)
    tracker.log_metrics("show-1", step=1, metrics={"loss": 0.7})
    tracker.log_metrics("show-2", step=1, metrics={"loss": 0.3})

    # Test show subcommand
    exit_code = main(["--data-root", str(tmp_path), "experiment", "show", "show-1"])
    show_output = capsys.readouterr().out

    assert exit_code == 0
    show_data = json.loads(show_output)
    assert show_data["run_id"] == "show-1"
    assert show_data["loss_final"] == 0.7

    # Test compare subcommand
    exit_code = main([
        "--data-root", str(tmp_path),
        "experiment", "compare", "show-1", "show-2",
    ])
    compare_output = capsys.readouterr().out

    assert exit_code == 0
    compare_data = json.loads(compare_output)
    assert len(compare_data) == 2
    ids = [c["run_id"] for c in compare_data]
    assert "show-1" in ids
    assert "show-2" in ids
