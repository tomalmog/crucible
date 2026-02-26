"""Server command wiring for Forge CLI.

This module registers the 'server' subcommand and launches the
collaboration server via uvicorn.
"""

from __future__ import annotations

import argparse

from core.errors import ForgeDependencyError
from store.dataset_sdk import ForgeClient


def add_server_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the server subcommand on the CLI parser.

    Args:
        subparsers: Argparse subparsers object.
    """
    parser = subparsers.add_parser(
        "server",
        help="Start the Forge collaboration server",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )


def run_server_command(
    client: ForgeClient,
    args: argparse.Namespace,
) -> int:
    """Launch the collaboration server.

    Args:
        client: SDK client (unused but required by dispatch).
        args: Parsed CLI args with host and port.

    Returns:
        Exit code.

    Raises:
        ForgeDependencyError: If uvicorn is not installed.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ForgeDependencyError(
            "uvicorn is required to run the collaboration server. "
            "Install it with: pip install uvicorn"
        ) from exc

    from server.app import create_app

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)
    return 0
