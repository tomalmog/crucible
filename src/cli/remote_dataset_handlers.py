"""Handler functions for remote dataset CLI sub-subcommands.

Extracted from remote_command_handlers to keep file sizes under 300 lines.
Each handler maps a parsed CLI namespace to a remote dataset operation.
"""

from __future__ import annotations

import argparse

from store.dataset_sdk import ForgeClient


def _handle_dataset_push(client: ForgeClient, args: argparse.Namespace) -> int:
    from serve.remote_dataset_ops import push_dataset
    from serve.ssh_connection import SshSession
    from store.cluster_registry import load_cluster

    cluster = load_cluster(client._config.data_root, args.cluster)
    with SshSession(cluster) as session:
        info = push_dataset(session, cluster, args.dataset, client._config.data_root)
    print(f"Pushed '{info.name}' ({_format_bytes(info.size_bytes)}) to {args.cluster}")
    return 0


def _handle_dataset_list(client: ForgeClient, args: argparse.Namespace) -> int:
    import json as json_mod
    from serve.remote_dataset_ops import list_remote_datasets
    from serve.ssh_connection import SshSession
    from store.cluster_registry import load_cluster

    cluster = load_cluster(client._config.data_root, args.cluster)
    with SshSession(cluster) as session:
        datasets = list_remote_datasets(session, cluster)
    if not datasets:
        print("No datasets on cluster.")
    else:
        for ds in datasets:
            print(f"  {ds.name}  {_format_bytes(ds.size_bytes)}  synced {ds.synced_at}")
    # Machine-readable output for Tauri
    print("FORGE_JSON:" + json_mod.dumps([
        {"name": d.name, "size_bytes": d.size_bytes,
         "version_id": d.version_id, "synced_at": d.synced_at}
        for d in datasets
    ]))
    return 0


def _handle_dataset_pull(client: ForgeClient, args: argparse.Namespace) -> int:
    from serve.remote_dataset_ops import pull_remote_dataset
    from serve.ssh_connection import SshSession
    from store.cluster_registry import load_cluster

    cluster = load_cluster(client._config.data_root, args.cluster)
    with SshSession(cluster) as session:
        local_path = pull_remote_dataset(
            session, cluster, args.dataset, client._config.data_root,
        )
    print(f"Pulled '{args.dataset}' to {local_path}")
    return 0


def _handle_dataset_delete(client: ForgeClient, args: argparse.Namespace) -> int:
    from serve.remote_dataset_ops import delete_remote_dataset
    from serve.ssh_connection import SshSession
    from store.cluster_registry import load_cluster

    cluster = load_cluster(client._config.data_root, args.cluster)
    with SshSession(cluster) as session:
        delete_remote_dataset(session, cluster, args.dataset)
    print(f"Deleted '{args.dataset}' from {args.cluster}")
    return 0


def _format_bytes(size: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} TB"
