"""Experiment tracking for training runs.

This module provides metrics logging, hyperparameter tracking,
hardware info recording, run summaries, and cross-run comparison.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricEntry:
    """Single metric measurement at a training step.

    Attributes:
        step: Training step number.
        name: Metric name (e.g. 'loss', 'accuracy').
        value: Metric value.
        timestamp: Unix timestamp when recorded.
    """

    step: int
    name: str
    value: float
    timestamp: float


@dataclass
class ExperimentRun:
    """Aggregated data for one experiment run.

    Attributes:
        run_id: Unique run identifier.
        method: Training method used.
        dataset: Dataset name.
        status: Current run status.
        hyperparameters: Training configuration.
        hardware: Hardware info dict.
        metrics: All recorded metrics.
        start_time: Run start timestamp.
        end_time: Run end timestamp.
        duration_seconds: Total duration.
    """

    run_id: str
    method: str = ""
    dataset: str = ""
    status: str = "running"
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    metrics: list[MetricEntry] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0


class ExperimentTracker:
    """Track experiments with metrics, hyperparameters, and hardware info."""

    def __init__(self, data_root: Path) -> None:
        self._data_root = data_root
        self._metrics_dir = data_root / "runs"

    def log_metrics(self, run_id: str, step: int, metrics: dict[str, float]) -> None:
        """Append metrics for a run at a given step."""
        run_dir = self._metrics_dir / run_id / "metrics"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = run_dir / "metrics.jsonl"
        ts = time.time()
        with open(metrics_file, "a", encoding="utf-8") as fh:
            for name, value in metrics.items():
                entry = {"step": step, "name": name, "value": value, "timestamp": ts}
                fh.write(json.dumps(entry) + "\n")

    def log_hyperparameters(self, run_id: str, params: dict[str, Any]) -> None:
        """Store hyperparameters for a run."""
        run_dir = self._metrics_dir / run_id / "metrics"
        run_dir.mkdir(parents=True, exist_ok=True)
        params_file = run_dir / "hyperparameters.json"
        with open(params_file, "w", encoding="utf-8") as fh:
            json.dump(params, fh, indent=2, default=str)

    def log_hardware(self, run_id: str, hardware_info: dict[str, Any]) -> None:
        """Store hardware information for a run."""
        run_dir = self._metrics_dir / run_id / "metrics"
        run_dir.mkdir(parents=True, exist_ok=True)
        hw_file = run_dir / "hardware.json"
        with open(hw_file, "w", encoding="utf-8") as fh:
            json.dump(hardware_info, fh, indent=2, default=str)

    def get_run_metrics(self, run_id: str) -> list[MetricEntry]:
        """Load all metrics for a run."""
        metrics_file = self._metrics_dir / run_id / "metrics" / "metrics.jsonl"
        if not metrics_file.exists():
            return []
        entries: list[MetricEntry] = []
        with open(metrics_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                entries.append(MetricEntry(
                    step=obj["step"], name=obj["name"],
                    value=obj["value"], timestamp=obj["timestamp"],
                ))
        return entries

    def get_run_summary(self, run_id: str) -> dict[str, Any]:
        """Get aggregated summary for a run."""
        metrics = self.get_run_metrics(run_id)
        params = self._load_json(run_id, "hyperparameters.json")
        hardware = self._load_json(run_id, "hardware.json")
        metric_names = sorted(set(m.name for m in metrics))
        summary: dict[str, Any] = {
            "run_id": run_id,
            "hyperparameters": params,
            "hardware": hardware,
            "metric_names": metric_names,
        }
        for name in metric_names:
            values = [m.value for m in metrics if m.name == name]
            if values:
                summary[f"{name}_final"] = values[-1]
                summary[f"{name}_min"] = min(values)
                summary[f"{name}_max"] = max(values)
        return summary

    def compare_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        """Compare metrics across multiple runs."""
        return [self.get_run_summary(rid) for rid in run_ids]

    def list_runs(self) -> list[str]:
        """List all run IDs that have metrics data."""
        if not self._metrics_dir.exists():
            return []
        runs = []
        for d in sorted(self._metrics_dir.iterdir()):
            if d.is_dir() and (d / "metrics").is_dir():
                runs.append(d.name)
        return runs

    def delete_run_metrics(self, run_id: str) -> bool:
        """Delete metrics data for a run. Returns True if deleted."""
        import shutil
        metrics_dir = self._metrics_dir / run_id / "metrics"
        if metrics_dir.exists():
            shutil.rmtree(metrics_dir)
            return True
        return False

    def _load_json(self, run_id: str, filename: str) -> dict[str, Any]:
        """Load a JSON file from a run's metrics directory."""
        path = self._metrics_dir / run_id / "metrics" / filename
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
