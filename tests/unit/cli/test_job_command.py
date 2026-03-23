"""Unit tests for the job CLI command."""

from __future__ import annotations

import argparse

from cli.job_command import add_job_command


def test_job_command_registers_subcommands() -> None:
    """job command should have list, sync, cancel, logs, result subcommands."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_job_command(subparsers)

    # list (no args required)
    args = parser.parse_args(["job", "list"])
    assert args.job_action == "list"

    # sync requires --job-id
    args = parser.parse_args(["job", "sync", "--job-id", "job-test00000001"])
    assert args.job_action == "sync"
    assert args.job_id == "job-test00000001"

    # cancel requires --job-id
    args = parser.parse_args(["job", "cancel", "--job-id", "job-test00000001"])
    assert args.job_action == "cancel"

    # logs requires --job-id, optional --tail
    args = parser.parse_args(["job", "logs", "--job-id", "job-test00000001"])
    assert args.job_action == "logs"
    assert args.tail == 200

    args = parser.parse_args(["job", "logs", "--job-id", "x", "--tail", "50"])
    assert args.tail == 50

    # result requires --job-id
    args = parser.parse_args(["job", "result", "--job-id", "job-test00000001"])
    assert args.job_action == "result"
