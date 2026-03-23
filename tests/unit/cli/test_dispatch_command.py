"""Unit tests for the dispatch CLI command."""

from __future__ import annotations

import argparse
import json

from cli.dispatch_command import add_dispatch_command


def test_dispatch_command_registers() -> None:
    """dispatch subcommand should accept --spec."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_dispatch_command(subparsers)

    args = parser.parse_args(["dispatch", "--spec", '{"job_type":"sft"}'])
    assert args.spec == '{"job_type":"sft"}'


def test_dispatch_command_spec_required() -> None:
    """dispatch without --spec should error."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_dispatch_command(subparsers)

    import sys
    from io import StringIO
    import pytest

    with pytest.raises(SystemExit):
        parser.parse_args(["dispatch"])
