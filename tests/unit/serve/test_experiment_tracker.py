"""Tests for experiment tracker."""

from __future__ import annotations

import json
from pathlib import Path

from serve.experiment_tracker import ExperimentTracker, MetricEntry


def test_log_and_retrieve_metrics(tmp_path: Path) -> None:
    """Log metrics and retrieve them."""
    tracker = ExperimentTracker(tmp_path)
    tracker.log_metrics("run-1", step=0, metrics={"loss": 2.5, "accuracy": 0.1})
    tracker.log_metrics("run-1", step=1, metrics={"loss": 2.0, "accuracy": 0.3})
    entries = tracker.get_run_metrics("run-1")
    assert len(entries) == 4
    loss_entries = [e for e in entries if e.name == "loss"]
    assert len(loss_entries) == 2
    assert loss_entries[0].value == 2.5
    assert loss_entries[1].value == 2.0


def test_log_hyperparameters(tmp_path: Path) -> None:
    """Log and retrieve hyperparameters."""
    tracker = ExperimentTracker(tmp_path)
    tracker.log_hyperparameters("run-1", {"lr": 0.001, "epochs": 3})
    summary = tracker.get_run_summary("run-1")
    assert summary["hyperparameters"]["lr"] == 0.001


def test_log_hardware(tmp_path: Path) -> None:
    """Log and retrieve hardware info."""
    tracker = ExperimentTracker(tmp_path)
    tracker.log_hardware("run-1", {"gpu": "RTX 4090", "vram_gb": 24})
    summary = tracker.get_run_summary("run-1")
    assert summary["hardware"]["gpu"] == "RTX 4090"


def test_get_run_summary(tmp_path: Path) -> None:
    """Summary includes min/max/final metrics."""
    tracker = ExperimentTracker(tmp_path)
    tracker.log_metrics("run-1", step=0, metrics={"loss": 3.0})
    tracker.log_metrics("run-1", step=1, metrics={"loss": 2.0})
    tracker.log_metrics("run-1", step=2, metrics={"loss": 1.0})
    summary = tracker.get_run_summary("run-1")
    assert summary["loss_final"] == 1.0
    assert summary["loss_min"] == 1.0
    assert summary["loss_max"] == 3.0


def test_compare_runs(tmp_path: Path) -> None:
    """Compare multiple runs."""
    tracker = ExperimentTracker(tmp_path)
    tracker.log_metrics("run-1", step=0, metrics={"loss": 2.0})
    tracker.log_metrics("run-2", step=0, metrics={"loss": 1.5})
    result = tracker.compare_runs(["run-1", "run-2"])
    assert len(result) == 2
    assert result[0]["run_id"] == "run-1"
    assert result[1]["run_id"] == "run-2"


def test_list_runs(tmp_path: Path) -> None:
    """List runs with metrics."""
    tracker = ExperimentTracker(tmp_path)
    tracker.log_metrics("run-a", step=0, metrics={"loss": 1.0})
    tracker.log_metrics("run-b", step=0, metrics={"loss": 2.0})
    runs = tracker.list_runs()
    assert "run-a" in runs
    assert "run-b" in runs


def test_delete_run_metrics(tmp_path: Path) -> None:
    """Delete run metrics."""
    tracker = ExperimentTracker(tmp_path)
    tracker.log_metrics("run-1", step=0, metrics={"loss": 1.0})
    assert tracker.delete_run_metrics("run-1") is True
    assert tracker.get_run_metrics("run-1") == []


def test_delete_nonexistent_run(tmp_path: Path) -> None:
    """Delete nonexistent run returns False."""
    tracker = ExperimentTracker(tmp_path)
    assert tracker.delete_run_metrics("nonexistent") is False


def test_empty_run_metrics(tmp_path: Path) -> None:
    """No metrics returns empty list."""
    tracker = ExperimentTracker(tmp_path)
    assert tracker.get_run_metrics("nonexistent") == []
