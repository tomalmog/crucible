"""Unit tests for compare CLI command wiring."""

from __future__ import annotations

import pytest

from cli.main import build_parser


def test_compare_parser_requires_run_ids() -> None:
    """Compare command should fail when --run-a or --run-b is missing."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["compare", "--run-a", "run-001"])
    with pytest.raises(SystemExit):
        parser.parse_args(["compare", "--run-b", "run-002"])
    with pytest.raises(SystemExit):
        parser.parse_args(["compare"])


def test_compare_parser_accepts_both_runs() -> None:
    """Compare command should parse both run IDs correctly."""
    parser = build_parser()
    args = parser.parse_args([
        "compare",
        "--run-a", "run-001",
        "--run-b", "run-002",
    ])

    assert args.command == "compare"
    assert args.run_a == "run-001"
    assert args.run_b == "run-002"
