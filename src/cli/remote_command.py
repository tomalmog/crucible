"""CLI subcommand for remote Slurm cluster operations.

Provides ``forge remote`` with sub-subcommands for cluster management,
job submission, status monitoring, log streaming, and model pulling.
"""

from __future__ import annotations

import argparse
import sys

from core.errors import ForgeRemoteError
from store.dataset_sdk import ForgeClient


def add_remote_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``forge remote`` command and its sub-subcommands."""
    remote = subparsers.add_parser("remote", help="Remote Slurm cluster operations")
    sub = remote.add_subparsers(dest="remote_action", required=True)

    # register-cluster
    reg = sub.add_parser("register-cluster", help="Register a Slurm cluster")
    reg.add_argument("--name", required=True, help="Unique cluster name")
    reg.add_argument("--host", required=True, help="SSH hostname")
    reg.add_argument("--user", required=True, help="SSH username")
    reg.add_argument("--ssh-key", default="", help="Path to SSH private key")
    reg.add_argument("--password", default="", help="SSH password")
    reg.add_argument("--partition", default="", help="Default Slurm partition")
    reg.add_argument("--module-loads", default="", help="Comma-separated module load commands")
    reg.add_argument("--remote-workspace", default="/tmp/forge-jobs", help="Remote workspace path")
    reg.add_argument("--python-path", default="python3", help="Remote Python path")
    reg.add_argument("--validate", action="store_true", help="Validate after registration")

    # list-clusters
    sub.add_parser("list-clusters", help="List registered clusters")

    # validate-cluster
    vc = sub.add_parser("validate-cluster", help="Validate cluster readiness")
    vc.add_argument("--cluster", required=True, help="Cluster name")

    # remove-cluster
    rc = sub.add_parser("remove-cluster", help="Remove a cluster registration")
    rc.add_argument("--cluster", required=True, help="Cluster name")

    # reset-env
    re_ = sub.add_parser("reset-env", help="Remove forge conda env on a cluster")
    re_.add_argument("--cluster", required=True, help="Cluster name")

    # submit
    sm = sub.add_parser("submit", help="Submit a remote training job")
    _add_submit_args(sm)

    # submit-sweep
    ss = sub.add_parser("submit-sweep", help="Submit a remote sweep")
    _add_submit_args(ss)
    ss.add_argument("--sweep-config", required=True, help="Path to sweep config YAML")

    # list (remote jobs)
    sub.add_parser("list", help="List remote jobs")

    # status
    st = sub.add_parser("status", help="Check remote job status")
    st.add_argument("--job-id", required=True, help="Remote job ID")

    # logs
    lg = sub.add_parser("logs", help="View remote job logs")
    lg.add_argument("--job-id", required=True, help="Remote job ID")
    lg.add_argument("--follow", action="store_true", help="Stream logs in real time")
    lg.add_argument("--tail", type=int, default=100, help="Number of trailing lines")

    # cancel
    cn = sub.add_parser("cancel", help="Cancel a remote job")
    cn.add_argument("--job-id", required=True, help="Remote job ID")

    # pull-model
    pm = sub.add_parser("pull-model", help="Download model from remote job")
    pm.add_argument("--job-id", required=True, help="Remote job ID")
    pm.add_argument("--model-name", default=None, help="Name to register model under (auto-generated if omitted)")

    # dataset-push
    dp = sub.add_parser("dataset-push", help="Push a dataset to a remote cluster")
    dp.add_argument("--cluster", required=True, help="Cluster name")
    dp.add_argument("--dataset", required=True, help="Dataset name")

    # dataset-list
    dl = sub.add_parser("dataset-list", help="List datasets on a remote cluster")
    dl.add_argument("--cluster", required=True, help="Cluster name")

    # dataset-pull
    dpl = sub.add_parser("dataset-pull", help="Pull a dataset from a remote cluster")
    dpl.add_argument("--cluster", required=True, help="Cluster name")
    dpl.add_argument("--dataset", required=True, help="Dataset name")

    # dataset-delete
    dd = sub.add_parser("dataset-delete", help="Delete a dataset on a remote cluster")
    dd.add_argument("--cluster", required=True, help="Cluster name")
    dd.add_argument("--dataset", required=True, help="Dataset name")


def _add_submit_args(parser: argparse.ArgumentParser) -> None:
    """Add common submission arguments to a parser."""
    parser.add_argument("--cluster", required=True, help="Cluster name")
    parser.add_argument("--method", required=True, help="Training method")
    parser.add_argument("--method-args", default="{}", help="JSON method arguments")
    parser.add_argument("--partition", default="", help="Slurm partition")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="GPUs per node")
    parser.add_argument("--gpu-type", default="", help="GPU type (e.g. a100)")
    parser.add_argument("--cpus-per-task", type=int, default=4, help="CPUs per task")
    parser.add_argument("--memory", default="32G", help="Memory limit")
    parser.add_argument("--time-limit", default="12:00:00", help="Wall-clock limit")
    parser.add_argument("--pull-model", action="store_true", help="Auto-pull model on completion")
    parser.add_argument("--model-name", default="", help="Name to register model under in registry")


def run_remote_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Dispatch to the appropriate remote sub-subcommand handler."""
    from cli.remote_command_handlers import (
        _handle_cancel,
        _handle_list_clusters,
        _handle_list_jobs,
        _handle_logs,
        _handle_pull_model,
        _handle_register_cluster,
        _handle_remove_cluster,
        _handle_reset_env,
        _handle_status,
        _handle_submit,
        _handle_submit_sweep,
        _handle_validate_cluster,
    )
    from cli.remote_dataset_handlers import (
        _handle_dataset_delete,
        _handle_dataset_list,
        _handle_dataset_pull,
        _handle_dataset_push,
    )

    handlers: dict[str, object] = {
        "register-cluster": _handle_register_cluster,
        "list-clusters": _handle_list_clusters,
        "validate-cluster": _handle_validate_cluster,
        "remove-cluster": _handle_remove_cluster,
        "reset-env": _handle_reset_env,
        "submit": _handle_submit,
        "submit-sweep": _handle_submit_sweep,
        "list": _handle_list_jobs,
        "status": _handle_status,
        "logs": _handle_logs,
        "cancel": _handle_cancel,
        "pull-model": _handle_pull_model,
        "dataset-push": _handle_dataset_push,
        "dataset-list": _handle_dataset_list,
        "dataset-pull": _handle_dataset_pull,
        "dataset-delete": _handle_dataset_delete,
    }
    handler = handlers.get(args.remote_action)
    if handler is None:
        print(f"Unknown remote action: {args.remote_action}", file=sys.stderr)
        return 2
    try:
        return handler(client, args)  # type: ignore[operator]
    except ForgeRemoteError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
