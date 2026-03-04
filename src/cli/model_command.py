"""Model registry CLI command wiring for Forge.

This module isolates the model subcommand parser and execution
logic, mapping CLI arguments to ModelRegistry operations.
"""

from __future__ import annotations

import argparse
import subprocess

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
    _add_delete_subcommand(model_subs)


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


def _run_register(client: ForgeClient, args: argparse.Namespace) -> int:
    """Execute the model register subcommand.

    Args:
        client: SDK client instance.
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    registry = client.model_registry()
    version = registry.register_model(args.name, args.model_path)
    if args.tag:
        registry.tag_version(version.version_id, args.tag)
        print(f"version_id={version.version_id}")
        print(f"tag={args.tag}")
    else:
        print(f"version_id={version.version_id}")
    return 0


def _run_list(client: ForgeClient, args: argparse.Namespace) -> int:
    """Execute the model list subcommand.

    Args:
        client: SDK client instance.
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    registry = client.model_registry()

    if args.name:
        # List versions of a specific model
        versions = registry.list_versions_for_model(args.name)
        if not versions:
            print(f"No versions registered for model '{args.name}'.")
            return 0
        active_id = registry.get_active_version_id_for_model(args.name)
        for v in versions:
            marker = " [active]" if v.version_id == active_id else ""
            run_info = v.run_id or "-"
            parent = v.parent_version_id or "-"
            print(
                f"{v.version_id}\t{v.model_path}\t"
                f"{run_info}\t{parent}\t{v.created_at}{marker}"
            )
        return 0

    # List all model names with version counts
    names = registry.list_model_names()
    if not names:
        print("No models registered.")
        return 0
    for name in names:
        versions = registry.list_versions_for_model(name)
        active_id = registry.get_active_version_id_for_model(name)
        active_label = f" (active: {active_id})" if active_id else ""
        print(f"{name}\t{len(versions)} version(s){active_label}")
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
    version = registry.rollback_to_version(args.name, args.version_id)
    print(f"Rolled back to version {version.version_id}")
    return 0


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


def _run_delete(client: ForgeClient, args: argparse.Namespace) -> int:
    """Execute the model delete subcommand."""
    registry = client.model_registry()

    if args.version_id:
        try:
            version = registry.get_version(args.version_id)
        except ForgeModelRegistryError as exc:
            print(f"Error: {exc}")
            return 1
        versions = [version]
    else:
        versions = list(registry.list_versions_for_model(args.name))
        if not versions:
            print(f"No versions found for model '{args.name}'.")
            return 1

    scope = f"version {args.version_id}" if args.version_id else f"model '{args.name}' ({len(versions)} version(s))"
    parts: list[str] = []
    if not args.keep_registry:
        parts.append("registry")
    if args.delete_local:
        parts.append("local files")
    if args.include_remote:
        parts.append("remote files")
    print(f"Will delete {scope} ({', '.join(parts)}):")
    for v in versions:
        path_info = v.model_path or "(no local path)"
        local_tag = " [will delete]" if args.delete_local and v.model_path else ""
        print(f"  {v.version_id}  {path_info}{local_tag}")
    if args.include_remote:
        for v in versions:
            if v.remote_path:
                print(f"  [remote] {v.remote_host}:{v.remote_path}")

    if not args.yes:
        answer = input("Delete? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return 0

    if not args.keep_registry:
        if args.version_id:
            result = registry.delete_version(
                args.name, args.version_id, delete_local=args.delete_local,
            )
        else:
            result = registry.delete_model(args.name, delete_local=args.delete_local)
        print(f"Removed {result.versions_removed} version(s).")
        if result.local_paths_deleted:
            for p in result.local_paths_deleted:
                print(f"  Deleted: {p}")
        if result.local_paths_skipped:
            for p in result.local_paths_skipped:
                print(f"  Skipped: {p}")
        for err in result.errors:
            print(f"  Warning: {err}")

    if args.include_remote:
        _delete_remote_paths(versions)

    return 0


def _delete_remote_paths(versions: list[object]) -> None:
    """Delete remote model files via SSH for eligible versions."""
    from core.model_registry_types import ModelVersion

    for v in versions:
        if not isinstance(v, ModelVersion):
            continue
        if not v.remote_host or not v.remote_path:
            continue
        if "/forge-jobs/rj-" not in v.remote_path:
            print(f"  Skipped remote (not a forge job path): {v.remote_path}")
            continue
        try:
            subprocess.run(
                ["ssh", v.remote_host, "rm", "-rf", v.remote_path],
                check=True, timeout=30,
            )
            print(f"  Deleted remote: {v.remote_host}:{v.remote_path}")
        except Exception as exc:
            print(f"  Failed remote delete {v.remote_host}:{v.remote_path}: {exc}")
