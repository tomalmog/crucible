"""Compute CLI command wiring for Forge.

This module isolates compute subcommand parser and execution logic,
mapping CLI arguments to local executor operations for job management.
"""

from __future__ import annotations

import argparse

from compute.compute_target import resolve_executor
from core.compute_types import ComputeTarget, JobSubmission
from store.dataset_sdk import ForgeClient


def add_compute_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register compute subcommand with sub-subcommands.

    Args:
        subparsers: Argparse subparsers object.
    """
    compute_parser = subparsers.add_parser(
        "compute",
        help="Compute job submission and management",
    )
    compute_subs = compute_parser.add_subparsers(
        dest="compute_action",
        required=True,
    )
    _add_submit_subcommand(compute_subs)
    _add_status_subcommand(compute_subs)


def run_compute_command(
    client: ForgeClient, args: argparse.Namespace,
) -> int:
    """Dispatch compute sub-subcommand execution.

    Args:
        client: SDK client instance.
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    action = args.compute_action
    if action == "submit":
        return _run_submit(args)
    if action == "status":
        return _run_status(args)
    return 2


def _add_submit_subcommand(compute_subs: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the submit sub-subcommand."""
    submit_parser = compute_subs.add_parser(
        "submit",
        help="Submit a compute job",
    )
    submit_parser.add_argument(
        "--command", required=True, dest="run_command",
        help="Command to execute",
    )
    submit_parser.add_argument(
        "--working-dir", default=None,
        help="Working directory for job execution",
    )


def _add_status_subcommand(compute_subs: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the status sub-subcommand."""
    status_parser = compute_subs.add_parser(
        "status",
        help="Check compute job status",
    )
    status_parser.add_argument(
        "--job-id", required=True,
        help="Job ID to query",
    )


def _run_submit(args: argparse.Namespace) -> int:
    """Execute the compute submit subcommand.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    target = ComputeTarget(name="local", executor_type="local")
    submission = JobSubmission(
        target=target,
        command=args.run_command,
        working_dir=args.working_dir,
    )
    executor = resolve_executor(target.executor_type)
    job_id = executor.submit(submission)
    print(f"Submitted job: {job_id}")
    return 0


def _run_status(args: argparse.Namespace) -> int:
    """Execute the compute status subcommand.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    executor = resolve_executor("local")
    status = executor.status(args.job_id)
    print(f"Job {status.job_id}: {status.state}")
    if status.exit_code is not None:
        print(f"Exit code: {status.exit_code}")
    return 0
