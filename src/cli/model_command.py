"""Model registry CLI command wiring for Forge.

This module isolates the model subcommand parser and execution
logic, mapping CLI arguments to ModelRegistry operations.
"""

from __future__ import annotations

import argparse

from core.errors import ForgeModelRegistryError
from store.dataset_sdk import ForgeClient
from store.model_diff import format_model_diff


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


def run_model_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Dispatch model sub-subcommand execution.

    Args:
        client: SDK client instance.
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    action = args.model_action
    if action == "list":
        return _run_list(client)
    if action == "register":
        return _run_register(client, args)
    if action == "tag":
        return _run_tag(client, args)
    if action == "diff":
        return _run_diff(client, args)
    if action == "rollback":
        return _run_rollback(client, args)
    return 2


def _add_list_subcommand(model_subs: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the list sub-subcommand."""
    model_subs.add_parser("list", help="List all model versions")


def _add_register_subcommand(model_subs: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the register sub-subcommand."""
    reg_parser = model_subs.add_parser("register", help="Register a model version")
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
        "--version-id",
        required=True,
        help="Model version ID to roll back to",
    )


def _run_register(client: ForgeClient, args: argparse.Namespace) -> int:
    """Execute the model register subcommand.

    Args:
        client: SDK client instance.
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    registry = client.model_registry()
    version = registry.register_model(args.model_path)
    if args.tag:
        registry.tag_version(version.version_id, args.tag)
        print(f"version_id={version.version_id}")
        print(f"tag={args.tag}")
    else:
        print(f"version_id={version.version_id}")
    return 0


def _run_list(client: ForgeClient) -> int:
    """Execute the model list subcommand.

    Args:
        client: SDK client instance.

    Returns:
        Exit code.
    """
    registry = client.model_registry()
    versions = registry.list_versions()
    if not versions:
        print("No model versions registered.")
        return 0
    active_id = registry.get_active_version_id()
    for v in versions:
        marker = " [active]" if v.version_id == active_id else ""
        run_info = v.run_id or "-"
        parent = v.parent_version_id or "-"
        print(
            f"{v.version_id}\t{v.model_path}\t"
            f"{run_info}\t{parent}\t{v.created_at}{marker}"
        )
    return 0


def _run_tag(client: ForgeClient, args: argparse.Namespace) -> int:
    """Execute the model tag subcommand.

    Args:
        client: SDK client instance.
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    registry = client.model_registry()
    tag = registry.tag_version(args.version_id, args.tag)
    print(f"Tagged {tag.version_id} as '{tag.tag_name}'")
    return 0


def _run_diff(client: ForgeClient, args: argparse.Namespace) -> int:
    """Execute the model diff subcommand.

    Args:
        client: SDK client instance.
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    registry = client.model_registry()
    diff = registry.diff_versions(args.version_a, args.version_b)
    lines = format_model_diff(diff)
    for line in lines:
        print(line)
    return 0


def _run_rollback(client: ForgeClient, args: argparse.Namespace) -> int:
    """Execute the model rollback subcommand.

    Args:
        client: SDK client instance.
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    registry = client.model_registry()
    version = registry.rollback_to_version(args.version_id)
    print(f"Rolled back to version {version.version_id}")
    return 0
