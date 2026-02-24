"""Unit tests for export-spec CLI command registration."""

from __future__ import annotations

import argparse

from cli.export_spec_command import add_export_spec_command


def test_export_spec_command_registers_in_parser() -> None:
    """export-spec subcommand should accept --run-id and optional --output."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_export_spec_command(subparsers)

    args = parser.parse_args(["export-spec", "--run-id", "run-123"])
    assert args.run_id == "run-123"
    assert args.output is None


def test_export_spec_command_accepts_output_flag() -> None:
    """export-spec should accept --output for file path."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_export_spec_command(subparsers)

    args = parser.parse_args(["export-spec", "--run-id", "run-456", "--output", "/tmp/spec.yaml"])
    assert args.output == "/tmp/spec.yaml"
