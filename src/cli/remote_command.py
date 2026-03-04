"""CLI subcommand for remote Slurm cluster operations.

Provides ``forge remote`` with sub-subcommands for cluster management,
job submission, status monitoring, log streaming, and model pulling.
"""

from __future__ import annotations

import argparse
import json
import sys

from core.errors import ForgeRemoteError
from core.slurm_types import SlurmResourceConfig
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


def _add_submit_args(parser: argparse.ArgumentParser) -> None:
    """Add common submission arguments to a parser."""
    parser.add_argument("--cluster", required=True, help="Cluster name")
    parser.add_argument("--method", required=True, help="Training method")
    parser.add_argument("--method-args", default="{}", help="JSON method arguments")
    parser.add_argument("--dataset", default="", help="Dataset name")
    parser.add_argument("--partition", default="", help="Slurm partition")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="GPUs per node")
    parser.add_argument("--gpu-type", default="", help="GPU type (e.g. a100)")
    parser.add_argument("--cpus-per-task", type=int, default=4, help="CPUs per task")
    parser.add_argument("--memory", default="32G", help="Memory limit")
    parser.add_argument("--time-limit", default="12:00:00", help="Wall-clock limit")
    parser.add_argument("--data-strategy", default="shared", choices=["scp", "shared", "s3"])
    parser.add_argument("--pull-model", action="store_true", help="Auto-pull model on completion")
    parser.add_argument("--model-name", default="", help="Name to register model under in registry")


def run_remote_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Dispatch to the appropriate remote sub-subcommand handler."""
    handlers: dict[str, object] = {
        "register-cluster": _handle_register_cluster,
        "list-clusters": _handle_list_clusters,
        "validate-cluster": _handle_validate_cluster,
        "remove-cluster": _handle_remove_cluster,
        "submit": _handle_submit,
        "submit-sweep": _handle_submit_sweep,
        "list": _handle_list_jobs,
        "status": _handle_status,
        "logs": _handle_logs,
        "cancel": _handle_cancel,
        "pull-model": _handle_pull_model,
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


def _handle_register_cluster(
    client: ForgeClient, args: argparse.Namespace,
) -> int:
    from dataclasses import replace as dc_replace

    from core.slurm_types import ClusterConfig
    from serve.cluster_validator import update_cluster_validated, validate_cluster
    from store.cluster_registry import save_cluster

    module_loads = tuple(
        m.strip() for m in args.module_loads.split(",") if m.strip()
    ) if args.module_loads else ()

    cluster = ClusterConfig(
        name=args.name,
        host=args.host,
        user=args.user,
        ssh_key_path=args.ssh_key,
        password=args.password,
        default_partition=args.partition,
        module_loads=module_loads,
        remote_workspace=args.remote_workspace,
        python_path=args.python_path,
    )

    if args.validate:
        print(f"Validating cluster '{args.name}'...")
        result = validate_cluster(cluster)
        _print_validation(result)
        if result.python_ok and result.slurm_ok:
            cluster = update_cluster_validated(cluster)
            if result.partitions:
                cluster = dc_replace(cluster, partitions=result.partitions)
            if result.gpu_types:
                cluster = dc_replace(cluster, gpu_types=result.gpu_types)
        else:
            print("Warning: Cluster validation had issues.", file=sys.stderr)

    save_cluster(client._config.data_root, cluster)
    print(f"Cluster '{args.name}' registered.")
    return 0


def _handle_list_clusters(
    client: ForgeClient, _args: argparse.Namespace,
) -> int:
    from store.cluster_registry import list_clusters

    clusters = list_clusters(client._config.data_root)
    if not clusters:
        print("No clusters registered.")
        return 0
    for c in clusters:
        validated = "validated" if c.validated_at else "not validated"
        print(f"  {c.name}  {c.user}@{c.host}  [{validated}]")
    return 0


def _handle_validate_cluster(
    client: ForgeClient, args: argparse.Namespace,
) -> int:
    from serve.cluster_validator import update_cluster_validated, validate_cluster
    from store.cluster_registry import load_cluster, save_cluster

    cluster = load_cluster(client._config.data_root, args.cluster)
    print(f"Validating '{args.cluster}'...")
    result = validate_cluster(cluster)
    _print_validation(result)

    if result.python_ok and result.slurm_ok:
        updated = update_cluster_validated(cluster)
        save_cluster(client._config.data_root, updated)
    return 0


def _handle_remove_cluster(
    client: ForgeClient, args: argparse.Namespace,
) -> int:
    from store.cluster_registry import remove_cluster
    remove_cluster(client._config.data_root, args.cluster)
    print(f"Cluster '{args.cluster}' removed.")
    return 0


def _handle_submit(client: ForgeClient, args: argparse.Namespace) -> int:
    from serve.remote_job_submitter import submit_remote_job

    method_args = json.loads(args.method_args)
    dataset_path, data_strategy = _resolve_dataset(
        client, method_args, args.dataset, args.data_strategy,
    )
    resources = _build_resources(args)
    record = submit_remote_job(
        data_root=client._config.data_root,
        cluster_name=args.cluster,
        training_method=args.method,
        method_args=method_args,
        resources=resources,
        data_strategy=data_strategy,
        dataset_path=dataset_path,
        pull_model=args.pull_model,
        model_name=args.model_name,
    )
    print(f"Submitted job {record.job_id} (Slurm: {record.slurm_job_id})")
    return 0


def _handle_submit_sweep(client: ForgeClient, args: argparse.Namespace) -> int:
    import yaml

    from serve.remote_job_submitter import submit_remote_sweep

    with open(args.sweep_config) as f:
        sweep_config = yaml.safe_load(f)

    trial_configs = sweep_config.get("trials", [])
    if not trial_configs:
        print("Error: sweep config has no trials.", file=sys.stderr)
        return 1

    method_args = json.loads(args.method_args)
    dataset_path, data_strategy = _resolve_dataset(
        client, method_args, args.dataset, args.data_strategy,
    )
    resources = _build_resources(args)
    record = submit_remote_sweep(
        data_root=client._config.data_root,
        cluster_name=args.cluster,
        training_method=args.method,
        trial_configs=trial_configs,
        resources=resources,
        data_strategy=data_strategy,
        dataset_path=dataset_path,
    )
    print(
        f"Submitted sweep {record.job_id} "
        f"({record.sweep_array_size} trials, Slurm: {record.slurm_job_id})"
    )
    return 0


def _handle_list_jobs(
    client: ForgeClient, _args: argparse.Namespace,
) -> int:
    from store.remote_job_store import list_remote_jobs

    jobs = list_remote_jobs(client._config.data_root)
    if not jobs:
        print("No remote jobs.")
        return 0
    for j in jobs:
        sweep_tag = f" (sweep, {j.sweep_array_size} trials)" if j.is_sweep else ""
        print(
            f"  {j.job_id}  {j.training_method}  "
            f"{j.state}  cluster={j.cluster_name}{sweep_tag}"
        )
    return 0


def _handle_status(client: ForgeClient, args: argparse.Namespace) -> int:
    from serve.remote_log_streamer import check_remote_job_state
    from store.remote_job_store import load_remote_job

    state = check_remote_job_state(client._config.data_root, args.job_id)
    record = load_remote_job(client._config.data_root, args.job_id)
    print(f"Job {record.job_id} (Slurm {record.slurm_job_id}): {state}")
    print(f"  Cluster: {record.cluster_name}")
    print(f"  Method: {record.training_method}")
    print(f"  Submitted: {record.submitted_at}")
    if record.model_path_remote:
        print(f"  Remote model: {record.model_path_remote}")
    if record.model_path_local:
        print(f"  Local model: {record.model_path_local}")
    return 0


def _handle_logs(client: ForgeClient, args: argparse.Namespace) -> int:
    if args.follow:
        from serve.remote_log_streamer import stream_remote_logs
        for line in stream_remote_logs(
            client._config.data_root, args.job_id, tail_lines=args.tail,
        ):
            print(line)
    else:
        from serve.remote_log_streamer import fetch_remote_logs
        content = fetch_remote_logs(
            client._config.data_root, args.job_id, tail_lines=args.tail,
        )
        print(content)
    return 0


def _handle_cancel(client: ForgeClient, args: argparse.Namespace) -> int:
    from serve.remote_job_submitter import cancel_remote_job
    record = cancel_remote_job(client._config.data_root, args.job_id)
    print(f"Cancelled job {record.job_id}.")
    return 0


def _handle_pull_model(client: ForgeClient, args: argparse.Namespace) -> int:
    from serve.remote_job_submitter import pull_remote_model
    record = pull_remote_model(
        client._config.data_root, args.job_id, args.model_name,
    )
    print(f"Model pulled to {record.model_path_local}")
    print(f"Registered as {record.local_version_id}")
    return 0


def _resolve_dataset(
    client: ForgeClient,
    method_args: dict[str, object],
    cli_dataset: str,
    data_strategy: str,
) -> tuple[str, str]:
    """Resolve a dataset name to a local path, auto-switching to scp if needed.

    Returns (dataset_path, data_strategy) with resolved values.
    """
    from pathlib import Path

    dataset_path = cli_dataset
    if not dataset_path:
        ds_name = str(method_args.get("dataset_name", ""))
        if ds_name:
            resolved = client.resolve_dataset_source(ds_name)
            if resolved:
                dataset_path = resolved
                print(f"Resolved dataset '{ds_name}' → {resolved}")

    if dataset_path and data_strategy == "shared":
        p = Path(dataset_path).expanduser().resolve()
        if p.exists():
            data_strategy = "scp"
            print("Local dataset detected — switching to scp upload strategy")

    return dataset_path, data_strategy


def _build_resources(args: argparse.Namespace) -> SlurmResourceConfig:
    """Build SlurmResourceConfig from parsed CLI arguments."""
    return SlurmResourceConfig(
        partition=args.partition,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        gpu_type=args.gpu_type,
        cpus_per_task=args.cpus_per_task,
        memory=args.memory,
        time_limit=args.time_limit,
    )


def _print_validation(result: object) -> None:
    """Print cluster validation results."""
    from core.slurm_types import ClusterValidationResult
    if not isinstance(result, ClusterValidationResult):
        return
    print(f"  Python: {'OK' if result.python_ok else 'MISSING'} {result.python_version}")
    print(f"  PyTorch: {'OK' if result.torch_ok else 'MISSING'} {result.torch_version}")
    print(f"  CUDA: {'OK' if result.cuda_ok else 'MISSING'} {result.cuda_version}")
    print(f"  Slurm: {'OK' if result.slurm_ok else 'MISSING'}")
    if result.partitions:
        print(f"  Partitions: {', '.join(result.partitions)}")
    if result.gpu_types:
        print(f"  GPU types: {', '.join(result.gpu_types)}")
    for error in result.errors:
        print(f"  ERROR: {error}")
