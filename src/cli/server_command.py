"""Server command wiring for Crucible CLI.

This module registers the 'server' subcommand and launches the
collaboration server via uvicorn.
"""

from __future__ import annotations

import argparse

from core.errors import CrucibleDependencyError
from store.dataset_sdk import CrucibleClient


def add_server_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the server subcommand on the CLI parser.

    Args:
        subparsers: Argparse subparsers object.
    """
    parser = subparsers.add_parser(
        "server",
        help="Start the Crucible collaboration server",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--token", type=str, default="",
        help="API bearer token for job endpoints (sets CRUCIBLE_API_TOKEN)",
    )


def run_server_command(
    client: CrucibleClient,
    args: argparse.Namespace,
) -> int:
    """Launch the collaboration server.

    Args:
        client: SDK client (unused but required by dispatch).
        args: Parsed CLI args with host and port.

    Returns:
        Exit code.

    Raises:
        CrucibleDependencyError: If uvicorn is not installed.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise CrucibleDependencyError(
            "uvicorn is required to run the collaboration server. "
            "Install it with: pip install uvicorn"
        ) from exc

    import os
    from server.app import create_app

    if args.token:
        os.environ["CRUCIBLE_API_TOKEN"] = args.token

    data_root = str(client._config.data_root)
    app = create_app(data_root=data_root)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0
