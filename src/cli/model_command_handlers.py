"""Handler functions for model CLI sub-subcommands.

Each handler maps a parsed CLI namespace to the appropriate
ModelRegistry operation and prints results.
"""

from __future__ import annotations

import argparse
import subprocess

from core.errors import CrucibleModelRegistryError
from store.dataset_sdk import CrucibleClient


def _run_list(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Execute the model list subcommand."""
    registry = client.model_registry()
    models = registry.list_models()
    if not models:
        print("No models registered.")
        return 0
    for entry in models:
        run_info = entry.run_id or "-"
        loc = entry.location_type
        print(f"{entry.model_name}\t{entry.model_path}\t{run_info}\t{loc}\t{entry.created_at}")
    return 0


def _run_register(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Execute the model register subcommand."""
    registry = client.model_registry()
    entry = registry.register_model(args.name, args.model_path)
    print(f"model_name={entry.model_name}")
    return 0


def _run_delete(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Execute the model delete subcommand."""
    registry = client.model_registry()

    try:
        entry = registry.get_model(args.name)
    except CrucibleModelRegistryError as exc:
        print(f"Error: {exc}")
        return 1

    parts: list[str] = []
    if not args.keep_registry:
        parts.append("registry")
    if args.delete_local:
        parts.append("local files")
    if args.include_remote:
        parts.append("remote files")
    print(f"Will delete model '{args.name}' ({', '.join(parts)}):")
    path_info = entry.model_path or "(no local path)"
    local_tag = " [will delete]" if args.delete_local and entry.model_path else ""
    print(f"  {path_info}{local_tag}")
    if args.include_remote and entry.remote_path:
        print(f"  [remote] {entry.remote_host}:{entry.remote_path}")

    if not args.yes:
        answer = input("Delete? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return 0

    if not args.keep_registry:
        result = registry.delete_model(args.name, delete_local=args.delete_local)
        print(f"Removed {result.entries_removed} model(s).")
        if result.local_paths_deleted:
            for p in result.local_paths_deleted:
                print(f"  Deleted: {p}")
        if result.local_paths_skipped:
            for p in result.local_paths_skipped:
                print(f"  Skipped: {p}")
        for err in result.errors:
            print(f"  Warning: {err}")

    if args.include_remote:
        _delete_remote_paths(entry)

    return 0


def _run_remote_sizes(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Get sizes of remote models on a cluster via SSH."""
    import json as json_mod

    from serve.ssh_connection import SshSession
    from store.cluster_registry import load_cluster

    cluster = load_cluster(client._config.data_root, args.cluster)
    registry = client.model_registry()
    models = registry.list_models()

    # Collect remote models on this cluster
    remote_models = [
        m for m in models
        if m.remote_host == cluster.host and m.remote_path
    ]
    if not remote_models:
        print("CRUCIBLE_JSON:" + json_mod.dumps({}))
        return 0

    # Build a single du command for all paths
    paths = [m.remote_path for m in remote_models]
    du_args = " ".join(f"'{p}'" for p in paths)

    with SshSession(cluster) as session:
        stdout, _, _ = session.execute(
            f"du -sb {du_args} 2>/dev/null || true", timeout=30,
        )

    size_map: dict[str, int] = {}
    for line in stdout.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            size_map[parts[1].strip()] = int(parts[0].strip())

    result: dict[str, int] = {}
    for m in remote_models:
        if m.remote_path in size_map:
            result[m.model_name] = size_map[m.remote_path]

    print("CRUCIBLE_JSON:" + json_mod.dumps(result))
    return 0


def _run_pull(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Execute the model pull subcommand."""
    registry = client.model_registry()

    try:
        entry = registry.get_model(args.name)
    except CrucibleModelRegistryError as exc:
        print(f"Error: {exc}")
        return 1

    if entry.location_type not in ("remote", "both"):
        print(f"Model '{args.name}' is local-only, nothing to pull.")
        return 0

    if entry.run_id:
        # Job-based pull via remote pull-model
        from serve.remote_model_puller import pull_remote_model
        pull_remote_model(
            client._config.data_root, entry.run_id, entry.model_name,
        )
    else:
        # Direct pull via host + path
        from serve.remote_model_puller import pull_remote_model_direct
        pull_remote_model_direct(
            client._config.data_root,
            entry.model_name,
            entry.remote_host,
            entry.remote_path,
        )
    return 0


def _delete_remote_paths(entry: object) -> None:
    """Delete remote model files via SSH for eligible entries."""
    from core.model_registry_types import ModelEntry

    if not isinstance(entry, ModelEntry):
        return
    if not entry.remote_host or not entry.remote_path:
        return
    if "/crucible-jobs/rj-" not in entry.remote_path:
        print(f"  Skipped remote (not a crucible job path): {entry.remote_path}")
        return
    try:
        subprocess.run(
            ["ssh", entry.remote_host, "rm", "-rf", entry.remote_path],
            check=True, timeout=30,
        )
        print(f"  Deleted remote: {entry.remote_host}:{entry.remote_path}")
    except Exception as exc:
        print(f"  Failed remote delete {entry.remote_host}:{entry.remote_path}: {exc}")
