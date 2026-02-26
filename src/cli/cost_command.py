"""Cost tracking command wiring for Forge CLI."""

from __future__ import annotations

import argparse

from serve.cost_tracker import CostTracker
from store.dataset_sdk import ForgeClient


def run_cost_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle cost subcommand dispatch."""
    tracker = CostTracker(client._config.data_root)
    subcmd = args.cost_subcommand
    if subcmd == "summary":
        return _run_summary(tracker)
    if subcmd == "run":
        return _run_detail(tracker, args.run_id)
    return 1


def _run_summary(tracker: CostTracker) -> int:
    summary = tracker.get_project_summary()
    print(f"total_runs={summary.total_runs}")
    print(f"total_gpu_hours={summary.total_gpu_hours}")
    print(f"total_electricity_kwh={summary.total_electricity_kwh}")
    print(f"total_cost_usd=${summary.total_cost_usd}")
    for run in summary.runs:
        print(f"  run={run.run_id}  gpu_hours={run.gpu_hours}  cost=${run.total_cost_usd}")
    return 0


def _run_detail(tracker: CostTracker, run_id: str) -> int:
    cost = tracker.get_run_cost(run_id)
    if not cost:
        print(f"No cost data for run {run_id}")
        return 0
    print(f"run_id={cost.run_id}")
    print(f"gpu_hours={cost.gpu_hours}")
    print(f"gpu_type={cost.gpu_type}")
    print(f"electricity_kwh={cost.electricity_kwh}")
    print(f"electricity_cost=${cost.electricity_cost_usd}")
    print(f"cloud_cost=${cost.cloud_cost_usd}")
    print(f"total_cost=${cost.total_cost_usd}")
    return 0


def add_cost_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register cost subcommand."""
    parser = subparsers.add_parser("cost", help="Training cost tracking")
    sub = parser.add_subparsers(dest="cost_subcommand", required=True)
    sub.add_parser("summary", help="Project cost summary")
    run_p = sub.add_parser("run", help="Per-run cost breakdown")
    run_p.add_argument("run_id", help="Run ID")
