"""Cloud burst command wiring for Forge CLI."""

from __future__ import annotations

import argparse
from typing import Any

from serve.cloud_burst import (
    estimate_cloud_cost,
    poll_cloud_job,
    submit_cloud_job,
    sync_cloud_results,
)
from store.dataset_sdk import ForgeClient


def run_cloud_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle cloud subcommand dispatch."""
    subcmd = args.cloud_subcommand
    if subcmd == "estimate":
        return _run_estimate(args)
    if subcmd == "submit":
        return _run_submit(args)
    if subcmd == "status":
        return _run_status(args)
    if subcmd == "sync":
        return _run_sync(client, args)
    return 1


def _run_estimate(args: argparse.Namespace) -> int:
    est = estimate_cloud_cost(args.hours, args.provider, args.gpu_type)
    print(f"provider={est.provider}")
    print(f"gpu_type={est.gpu_type}")
    print(f"estimated_hours={est.estimated_hours}")
    print(f"cost_per_hour=${est.cost_per_hour}")
    print(f"total_cost=${est.total_cost}")
    return 0


def _run_submit(args: argparse.Namespace) -> int:
    status = submit_cloud_job({}, args.provider, args.api_key or "")
    print(f"job_id={status.job_id}")
    print(f"status={status.status}")
    return 0


def _run_status(args: argparse.Namespace) -> int:
    status = poll_cloud_job(args.job_id, args.provider)
    print(f"job_id={status.job_id}")
    print(f"status={status.status}")
    print(f"progress={status.progress}%")
    return 0


def _run_sync(client: ForgeClient, args: argparse.Namespace) -> int:
    path = sync_cloud_results(args.job_id, args.provider, client._config.data_root)
    print(f"synced_to={path}")
    return 0


def add_cloud_command(subparsers: Any) -> None:
    """Register cloud subcommand."""
    parser = subparsers.add_parser("cloud", help="Cloud GPU burst training")
    sub = parser.add_subparsers(dest="cloud_subcommand", required=True)

    est = sub.add_parser("estimate", help="Estimate cloud training cost")
    est.add_argument("--hours", type=float, required=True, help="Estimated training hours")
    est.add_argument("--provider", default="modal", choices=["modal", "runpod", "lambda"], help="Cloud provider")
    est.add_argument("--gpu-type", default="a100", help="GPU type")

    submit = sub.add_parser("submit", help="Submit cloud training job")
    submit.add_argument("--provider", default="modal", choices=["modal", "runpod", "lambda"], help="Cloud provider")
    submit.add_argument("--api-key", help="Provider API key")
    submit.add_argument("--config", help="Training config file path")

    status = sub.add_parser("status", help="Check cloud job status")
    status.add_argument("--job-id", required=True, help="Job ID to check")
    status.add_argument("--provider", default="modal", help="Cloud provider")

    sync = sub.add_parser("sync", help="Download cloud job results")
    sync.add_argument("--job-id", required=True, help="Job ID to sync")
    sync.add_argument("--provider", default="modal", help="Cloud provider")
