"""Unit tests for model CLI command wiring."""

from __future__ import annotations

import pytest

from cli.main import build_parser


def test_model_command_registers_in_parser() -> None:
    """Model subcommand should be registered and parseable."""
    parser = build_parser()
    args = parser.parse_args(["model", "list"])
    assert args.command == "model"
    assert args.model_action == "list"


