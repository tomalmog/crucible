"""Handler functions for remote CLI sub-subcommands.

Each handler maps a parsed CLI namespace to the appropriate
remote operation (cluster management, job submission, etc.)
and prints results.
"""

from __future__ import annotations

import argparse
import json

from store.dataset_sdk import CrucibleClient


def _handle_register_cluster(
    client: CrucibleClient, args: argparse.Namespace,
) -> int:
    from dataclasses import replace as dc_replace

    from core.slurm_types import ClusterConfig
    from serve.cluster_validator import update_cluster_validated, validate_cluster
    from store.cluster_registry import save_cluster

    from store.cluster_registry import load_cluster as _load_cluster

    module_loads = tuple(
        m.strip() for m in args.module_loads.split(",") if m.strip()
    ) if args.module_loads else ()

    # If cluster already exists, merge to preserve sensitive/validated fields
    existing: ClusterConfig | None = None
    try:
        existing = _load_cluster(client._config.data_root, args.name)
    except Exception:
        pass

    cluster = ClusterConfig(
        name=args.name,
        host=args.host,
        user=args.user,
        ssh_key_path=args.ssh_key or (existing.ssh_key_path if existing else ""),
        password=args.password or (existing.password if existing else ""),
        default_partition=args.partition,
        module_loads=module_loads,
        remote_workspace=args.remote_workspace,
        python_path=args.python_path,
        partitions=existing.partitions if existing else (),
        gpu_types=existing.gpu_types if existing else (),
        validated_at=existing.validated_at if existing else "",
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
            import sys
            print("Warning: Cluster validation had issues.", file=sys.stderr)

    save_cluster(client._config.data_root, cluster)
    print(f"Cluster '{args.name}' registered.")
    return 0


def _handle_list_clusters(
    client: CrucibleClient, _args: argparse.Namespace,
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
    client: CrucibleClient, args: argparse.Namespace,
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
    client: CrucibleClient, args: argparse.Namespace,
) -> int:
    from store.cluster_registry import remove_cluster
    remove_cluster(client._config.data_root, args.cluster)
    print(f"Cluster '{args.cluster}' removed.")
    return 0


def _handle_reset_env(
    client: CrucibleClient, args: argparse.Namespace,
) -> int:
    from serve.remote_env_setup import reset_remote_env
    from serve.ssh_connection import SshSession
    from store.cluster_registry import load_cluster

    cluster = load_cluster(client._config.data_root, args.cluster)
    with SshSession(cluster) as session:
        reset_remote_env(session)
    return 0


def _handle_submit(client: CrucibleClient, args: argparse.Namespace) -> int:
    from serve.remote_job_submitter import submit_remote_job

    method_args = json.loads(args.method_args)
    resources = _build_resources(args)
    record = submit_remote_job(
        data_root=client._config.data_root,
        cluster_name=args.cluster,
        training_method=args.method,
        method_args=method_args,
        resources=resources,
        pull_model=args.pull_model,
        model_name=args.model_name,
    )
    print(f"Submitted job {record.job_id} (Slurm: {record.slurm_job_id})")
    return 0


def _handle_eval_submit(client: CrucibleClient, args: argparse.Namespace) -> int:
    from serve.remote_job_submitter import submit_remote_eval_job

    resources = _build_resources(args)
    method_args: dict[str, object] = {
        "model_path": args.model_path,
        "benchmarks": args.benchmarks,
    }
    if args.max_samples is not None:
        method_args["max_samples"] = args.max_samples
    if args.base_model:
        method_args["base_model_path"] = args.base_model
    record = submit_remote_eval_job(
        data_root=client._config.data_root,
        cluster_name=args.cluster,
        method_args=method_args,
        resources=resources,
        model_name=args.model_name,
    )
    print(f"job_id={record.job_id}")
    print(f"slurm_job_id={record.slurm_job_id}")
    return 0


def _handle_submit_sweep(client: CrucibleClient, args: argparse.Namespace) -> int:
    import yaml

    from serve.remote_job_submitter import submit_remote_sweep

    with open(args.sweep_config) as f:
        sweep_config = yaml.safe_load(f)

    trial_configs = sweep_config.get("trials", [])
    if not trial_configs:
        import sys
        print("Error: sweep config has no trials.", file=sys.stderr)
        return 1

    resources = _build_resources(args)
    record = submit_remote_sweep(
        data_root=client._config.data_root,
        cluster_name=args.cluster,
        training_method=args.method,
        trial_configs=trial_configs,
        resources=resources,
    )
    print(
        f"Submitted sweep {record.job_id} "
        f"({record.sweep_array_size} trials, Slurm: {record.slurm_job_id})"
    )
    return 0


def _handle_list_jobs(
    client: CrucibleClient, _args: argparse.Namespace,
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


def _handle_status(client: CrucibleClient, args: argparse.Namespace) -> int:
    from serve.remote_job_state import check_remote_job_state
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


def _handle_result(client: CrucibleClient, args: argparse.Namespace) -> int:
    from serve.remote_result_reader import read_remote_result
    from serve.ssh_connection import SshSession
    from store.cluster_registry import load_cluster
    from store.remote_job_store import load_remote_job

    record = load_remote_job(client._config.data_root, args.job_id)
    cluster = load_cluster(client._config.data_root, record.cluster_name)
    with SshSession(cluster) as session:
        result = read_remote_result(session, record)
    print(f"CRUCIBLE_JSON:{json.dumps(result)}")
    return 0


def _handle_logs(client: CrucibleClient, args: argparse.Namespace) -> int:
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


def _handle_cancel(client: CrucibleClient, args: argparse.Namespace) -> int:
    from serve.remote_model_puller import cancel_remote_job
    record = cancel_remote_job(client._config.data_root, args.job_id)
    print(f"Cancelled job {record.job_id}.")
    return 0


def _handle_pull_model(client: CrucibleClient, args: argparse.Namespace) -> int:
    from serve.remote_model_puller import pull_remote_model
    record = pull_remote_model(
        client._config.data_root, args.job_id, args.model_name,
    )
    print(f"Model pulled to {record.model_path_local}")
    print(f"Registered as {record.local_version_id}")
    return 0


def _handle_remote_chat(
    client: CrucibleClient, args: argparse.Namespace,
) -> int:
    """Stream chat inference from a remote cluster model."""
    import sys

    from core.chat_types import ChatOptions
    from core.slurm_types import SlurmResourceConfig
    from serve.remote_chat_runner import stream_remote_chat

    options = ChatOptions(
        model_path=args.model_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        stream=True,
    )
    resources = SlurmResourceConfig(
        partition=args.partition,
        gpu_type=args.gpu_type,
        memory=args.memory,
        time_limit=args.time_limit,
    )
    for chunk in stream_remote_chat(
        client._config.data_root, args.cluster, options, resources,
    ):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    return 0


def _build_resources(args: argparse.Namespace) -> "SlurmResourceConfig":
    """Build SlurmResourceConfig from parsed CLI arguments."""
    from core.slurm_types import SlurmResourceConfig

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
