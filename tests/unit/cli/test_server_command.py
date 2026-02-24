"""Unit tests for server CLI command wiring."""

from __future__ import annotations

from cli.main import build_parser


def test_server_command_registers_in_parser() -> None:
    """Server subcommand should be registered and parseable."""
    parser = build_parser()
    args = parser.parse_args(["server"])
    assert args.command == "server"
    assert args.port == 8080
    assert args.host == "0.0.0.0"


def test_server_command_accepts_custom_port_and_host() -> None:
    """Server subcommand should accept --port and --host options."""
    parser = build_parser()
    args = parser.parse_args([
        "server",
        "--port", "3000",
        "--host", "127.0.0.1",
    ])
    assert args.port == 3000
    assert args.host == "127.0.0.1"
