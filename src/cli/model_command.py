"""Model registry CLI command wiring for Crucible.

This module isolates the model subcommand parser and execution
logic, mapping CLI arguments to ModelRegistry operations.
"""

from __future__ import annotations

import argparse

from store.dataset_sdk import CrucibleClient


def add_model_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register model subcommand with sub-subcommands."""
    model_parser = subparsers.add_parser(
        "model",
        help="Model registry operations",
    )
    model_subs = model_parser.add_subparsers(
        dest="model_action",
        required=True,
    )
    _add_list_subcommand(model_subs)
    _add_register_subcommand(model_subs)
    _add_delete_subcommand(model_subs)
    _add_pull_subcommand(model_subs)
    _add_remote_sizes_subcommand(model_subs)


def run_model_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Dispatch model sub-subcommand execution."""
    from cli.model_command_handlers import (
        _run_delete,
        _run_list,
        _run_pull,
        _run_register,
        _run_remote_sizes,
    )

    action = args.model_action
    if action == "list":
        return _run_list(client, args)
    if action == "register":
        return _run_register(client, args)
    if action == "delete":
        return _run_delete(client, args)
    if action == "pull":
        return _run_pull(client, args)
    if action == "remote-sizes":
        return _run_remote_sizes(client, args)
    return 2


def _add_list_subcommand(model_subs: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    model_subs.add_parser("list", help="List registered models")


def _add_register_subcommand(model_subs: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    reg_parser = model_subs.add_parser("register", help="Register a model")
    reg_parser.add_argument(
        "--name", required=True, help="Model name to register",
    )
    reg_parser.add_argument(
        "--model-path", required=True, help="Path to model artifact",
    )


def _add_delete_subcommand(
    model_subs: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    del_parser = model_subs.add_parser("delete", help="Delete a model")
    del_parser.add_argument(
        "--name", required=True, help="Model name to delete",
    )
    del_parser.add_argument(
        "--delete-local", action="store_true",
        help="Also delete local model files on disk",
    )
    del_parser.add_argument(
        "--include-remote", action="store_true",
        help="Also delete remote model files via SSH",
    )
    del_parser.add_argument(
        "--keep-registry", action="store_true",
        help="Keep registry entries (only delete files)",
    )
    del_parser.add_argument(
        "--yes", action="store_true",
        help="Skip interactive confirmation",
    )


def _add_pull_subcommand(
    model_subs: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    pull_parser = model_subs.add_parser(
        "pull", help="Pull a remote model to local storage",
    )
    pull_parser.add_argument(
        "--name", required=True, help="Model name to pull",
    )


def _add_remote_sizes_subcommand(
    model_subs: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    sizes_parser = model_subs.add_parser(
        "remote-sizes", help="Get sizes of remote models on a cluster",
    )
    sizes_parser.add_argument(
        "--cluster", required=True, help="Cluster name",
    )
