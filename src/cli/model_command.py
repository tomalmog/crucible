"""Model registry CLI command wiring for Crucible.

This module isolates the model subcommand parser and execution
logic, mapping CLI arguments to ModelRegistry operations.
"""

from __future__ import annotations

import argparse

from store.dataset_sdk import CrucibleClient


def add_model_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register model subcommand with sub-subcommands.

    Args:
        subparsers: Argparse subparsers object.
    """
    model_parser = subparsers.add_parser(
        "model",
        help="Model versioning and registry operations",
    )
    model_subs = model_parser.add_subparsers(
        dest="model_action",
        required=True,
    )
    _add_list_subcommand(model_subs)
    _add_register_subcommand(model_subs)
    _add_tag_subcommand(model_subs)
    _add_diff_subcommand(model_subs)
    _add_rollback_subcommand(model_subs)
    _add_delete_subcommand(model_subs)


def run_model_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Dispatch model sub-subcommand execution.

    Args:
        client: SDK client instance.
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    from cli.model_command_handlers import (
        _run_delete,
        _run_diff,
        _run_list,
        _run_register,
        _run_rollback,
        _run_tag,
    )

    action = args.model_action
    if action == "list":
        return _run_list(client, args)
    if action == "register":
        return _run_register(client, args)
    if action == "tag":
        return _run_tag(client, args)
    if action == "diff":
        return _run_diff(client, args)
    if action == "rollback":
        return _run_rollback(client, args)
    if action == "delete":
        return _run_delete(client, args)
    return 2


def _add_list_subcommand(model_subs: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the list sub-subcommand."""
    list_parser = model_subs.add_parser("list", help="List models or model versions")
    list_parser.add_argument(
        "--name",
        default=None,
        help="List versions of a specific model (omit to list all model names)",
    )


def _add_register_subcommand(model_subs: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the register sub-subcommand."""
    reg_parser = model_subs.add_parser("register", help="Register a model version")
    reg_parser.add_argument(
        "--name",
        required=True,
        help="Model name to register version under",
    )
    reg_parser.add_argument(
        "--model-path",
        required=True,
        help="Path to model artifact",
    )
    reg_parser.add_argument(
        "--tag",
        default=None,
        help="Tag name for the registered version",
    )


def _add_tag_subcommand(model_subs: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the tag sub-subcommand."""
    tag_parser = model_subs.add_parser("tag", help="Tag a model version")
    tag_parser.add_argument(
        "--version-id",
        required=True,
        help="Model version ID to tag",
    )
    tag_parser.add_argument(
        "--tag",
        required=True,
        help="Tag name to assign",
    )


def _add_diff_subcommand(model_subs: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the diff sub-subcommand."""
    diff_parser = model_subs.add_parser(
        "diff",
        help="Compare two model versions",
    )
    diff_parser.add_argument(
        "--version-a",
        required=True,
        help="First model version ID",
    )
    diff_parser.add_argument(
        "--version-b",
        required=True,
        help="Second model version ID",
    )


def _add_rollback_subcommand(model_subs: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the rollback sub-subcommand."""
    rb_parser = model_subs.add_parser(
        "rollback",
        help="Rollback to a model version",
    )
    rb_parser.add_argument(
        "--name",
        required=True,
        help="Model name to rollback within",
    )
    rb_parser.add_argument(
        "--version-id",
        required=True,
        help="Model version ID to roll back to",
    )


def _add_delete_subcommand(
    model_subs: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the delete sub-subcommand."""
    del_parser = model_subs.add_parser(
        "delete",
        help="Delete a model or a single version",
    )
    del_parser.add_argument(
        "--name", required=True, help="Model name to delete",
    )
    del_parser.add_argument(
        "--version-id", default=None,
        help="Delete only this version (omit to delete entire model)",
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
