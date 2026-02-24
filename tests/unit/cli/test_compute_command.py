"""Unit tests for compute CLI command wiring."""

from __future__ import annotations

import pytest

from cli.main import build_parser


def test_compute_command_registers_in_parser() -> None:
    """Compute subcommand should be registered and parseable."""
    parser = build_parser()
    args = parser.parse_args([
        "compute", "submit",
        "--command", "echo",
    ])
    assert args.command == "compute"
    assert args.compute_action == "submit"
    assert args.run_command == "echo"


def test_compute_submit_parses_all_args() -> None:
    """Compute submit should accept command and working-dir args."""
    parser = build_parser()
    args = parser.parse_args([
        "compute", "submit",
        "--command", "python train.py",
        "--working-dir", "/tmp/work",
    ])
    assert args.command == "compute"
    assert args.compute_action == "submit"
    assert args.run_command == "python train.py"
    assert args.working_dir == "/tmp/work"


def test_compute_status_parses_job_id() -> None:
    """Compute status should accept a --job-id argument."""
    parser = build_parser()
    args = parser.parse_args([
        "compute", "status",
        "--job-id", "abc123",
    ])
    assert args.command == "compute"
    assert args.compute_action == "status"
    assert args.job_id == "abc123"
