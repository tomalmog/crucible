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


def test_model_tag_subcommand_parses_args() -> None:
    """Model tag subcommand should accept version-id and tag args."""
    parser = build_parser()
    args = parser.parse_args([
        "model", "tag",
        "--version-id", "mv-abc123",
        "--tag", "production",
    ])
    assert args.command == "model"
    assert args.model_action == "tag"
    assert args.version_id == "mv-abc123"
    assert args.tag == "production"


def test_model_diff_subcommand_parses_args() -> None:
    """Model diff subcommand should accept version-a and version-b."""
    parser = build_parser()
    args = parser.parse_args([
        "model", "diff",
        "--version-a", "mv-aaa",
        "--version-b", "mv-bbb",
    ])
    assert args.command == "model"
    assert args.model_action == "diff"
    assert args.version_a == "mv-aaa"
    assert args.version_b == "mv-bbb"
