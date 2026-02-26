"""Experiment tracking command wiring for Forge CLI.

This module provides commands for listing, viewing, comparing,
and deleting experiment run metrics.
"""

from __future__ import annotations

import argparse
import json

from serve.experiment_tracker import ExperimentTracker
from store.dataset_sdk import ForgeClient


def run_experiment_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle experiment subcommand dispatch."""
    tracker = ExperimentTracker(client._config.data_root)
    subcmd = args.experiment_subcommand
    if subcmd == "list":
        return _run_list(tracker)
    if subcmd == "show":
        return _run_show(tracker, args.run_id)
    if subcmd == "compare":
        return _run_compare(tracker, args.run_ids)
    if subcmd == "delete":
        return _run_delete(tracker, args.run_id)
    print(f"Unknown experiment subcommand: {subcmd}")
    return 1


def _run_list(tracker: ExperimentTracker) -> int:
    """List all experiment runs."""
    runs = tracker.list_runs()
    if not runs:
        print("No experiment runs found.")
        return 0
    print(f"Found {len(runs)} experiment run(s):")
    for run_id in runs:
        summary = tracker.get_run_summary(run_id)
        loss_final = summary.get("loss_final", "-")
        print(f"  {run_id}  loss={loss_final}")
    return 0


def _run_show(tracker: ExperimentTracker, run_id: str) -> int:
    """Show detailed metrics for a run."""
    summary = tracker.get_run_summary(run_id)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _run_compare(tracker: ExperimentTracker, run_ids: list[str]) -> int:
    """Compare metrics across runs."""
    comparisons = tracker.compare_runs(run_ids)
    print(json.dumps(comparisons, indent=2, default=str))
    return 0


def _run_delete(tracker: ExperimentTracker, run_id: str) -> int:
    """Delete metrics for a run."""
    deleted = tracker.delete_run_metrics(run_id)
    if deleted:
        print(f"Deleted metrics for run {run_id}")
    else:
        print(f"No metrics found for run {run_id}")
    return 0


def add_experiment_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register experiment subcommand with sub-subcommands."""
    parser = subparsers.add_parser("experiment", help="Experiment tracking and comparison")
    sub = parser.add_subparsers(dest="experiment_subcommand", required=True)
    sub.add_parser("list", help="List all experiment runs")
    show_p = sub.add_parser("show", help="Show detailed run metrics")
    show_p.add_argument("run_id", help="Run ID to show")
    compare_p = sub.add_parser("compare", help="Compare multiple runs")
    compare_p.add_argument("run_ids", nargs="+", help="Run IDs to compare")
    delete_p = sub.add_parser("delete", help="Delete run metrics")
    delete_p.add_argument("run_id", help="Run ID to delete")
