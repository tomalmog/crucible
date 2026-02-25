"""Training cost and resource tracking.

This module tracks GPU-hours, estimated electricity costs,
and aggregate resource usage per training run and project.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunCost:
    """Cost breakdown for a single training run.

    Attributes:
        run_id: Training run identifier.
        gpu_hours: Total GPU-hours consumed.
        gpu_type: GPU model used.
        tdp_watts: GPU thermal design power.
        electricity_kwh: Estimated electricity usage.
        electricity_cost_usd: Estimated electricity cost.
        cloud_cost_usd: Cloud compute cost (if applicable).
        total_cost_usd: Total estimated cost.
    """

    run_id: str
    gpu_hours: float = 0.0
    gpu_type: str = "unknown"
    tdp_watts: int = 300
    electricity_kwh: float = 0.0
    electricity_cost_usd: float = 0.0
    cloud_cost_usd: float = 0.0
    total_cost_usd: float = 0.0


@dataclass(frozen=True)
class ProjectCostSummary:
    """Aggregate cost summary across all runs in a project.

    Attributes:
        total_runs: Number of training runs.
        total_gpu_hours: Total GPU-hours across all runs.
        total_electricity_kwh: Total electricity usage.
        total_cost_usd: Total estimated cost.
        runs: Per-run cost breakdowns.
    """

    total_runs: int
    total_gpu_hours: float
    total_electricity_kwh: float
    total_cost_usd: float
    runs: tuple[RunCost, ...] = ()


class CostTracker:
    """Track training costs and resource usage."""

    def __init__(self, data_root: Path) -> None:
        self._data_root = data_root
        self._cost_dir = data_root / "costs"

    def log_run_cost(
        self,
        run_id: str,
        duration_seconds: float,
        gpu_type: str = "unknown",
        tdp_watts: int = 300,
        electricity_rate_per_kwh: float = 0.12,
        cloud_cost_usd: float = 0.0,
    ) -> RunCost:
        """Log cost for a training run."""
        gpu_hours = duration_seconds / 3600.0
        electricity_kwh = (tdp_watts / 1000.0) * gpu_hours
        electricity_cost = electricity_kwh * electricity_rate_per_kwh
        total = electricity_cost + cloud_cost_usd
        cost = RunCost(
            run_id=run_id,
            gpu_hours=round(gpu_hours, 4),
            gpu_type=gpu_type,
            tdp_watts=tdp_watts,
            electricity_kwh=round(electricity_kwh, 4),
            electricity_cost_usd=round(electricity_cost, 4),
            cloud_cost_usd=cloud_cost_usd,
            total_cost_usd=round(total, 4),
        )
        self._store_cost(cost)
        return cost

    def get_run_cost(self, run_id: str) -> RunCost | None:
        """Get cost data for a specific run."""
        path = self._cost_dir / f"{run_id}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return RunCost(**data)

    def get_project_summary(self) -> ProjectCostSummary:
        """Get aggregate cost summary for the project."""
        if not self._cost_dir.exists():
            return ProjectCostSummary(
                total_runs=0, total_gpu_hours=0.0,
                total_electricity_kwh=0.0, total_cost_usd=0.0,
            )
        runs: list[RunCost] = []
        for f in sorted(self._cost_dir.glob("*.json")):
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            runs.append(RunCost(**data))
        return ProjectCostSummary(
            total_runs=len(runs),
            total_gpu_hours=round(sum(r.gpu_hours for r in runs), 4),
            total_electricity_kwh=round(sum(r.electricity_kwh for r in runs), 4),
            total_cost_usd=round(sum(r.total_cost_usd for r in runs), 4),
            runs=tuple(runs),
        )

    def _store_cost(self, cost: RunCost) -> None:
        """Persist cost data to disk."""
        self._cost_dir.mkdir(parents=True, exist_ok=True)
        path = self._cost_dir / f"{cost.run_id}.json"
        data = {
            "run_id": cost.run_id, "gpu_hours": cost.gpu_hours,
            "gpu_type": cost.gpu_type, "tdp_watts": cost.tdp_watts,
            "electricity_kwh": cost.electricity_kwh,
            "electricity_cost_usd": cost.electricity_cost_usd,
            "cloud_cost_usd": cost.cloud_cost_usd,
            "total_cost_usd": cost.total_cost_usd,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
