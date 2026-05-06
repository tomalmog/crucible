"""Shared helpers for remote Python runtime selection."""

from __future__ import annotations

import shlex
from dataclasses import replace

from core.slurm_types import ClusterConfig, SlurmResourceConfig
from serve.managed_conda_env import (
    managed_conda_activate,
    managed_conda_bootstrap,
)
from serve.ssh_connection import SshSession

_MANAGED_PYTHON_PATHS = frozenset({"python", "python3"})
_MANAGED_ENV_PYTHON = '"${CONDA_PREFIX:?managed crucible env inactive}/bin/python"'


def uses_managed_conda_env(cluster: ClusterConfig) -> bool:
    """Return whether the cluster should use Crucible's managed conda env."""
    return cluster.python_path.strip() in _MANAGED_PYTHON_PATHS


def resolve_cluster_runtime(
    cluster: ClusterConfig,
    session: SshSession,
) -> ClusterConfig:
    """Resolve ``~`` in remote workspace and explicit Python paths."""
    resolved_workspace = session.resolve_path(cluster.remote_workspace)
    resolved_python = cluster.python_path
    if not uses_managed_conda_env(cluster):
        resolved_python = session.resolve_path(cluster.python_path)
    if (
        resolved_workspace == cluster.remote_workspace
        and resolved_python == cluster.python_path
    ):
        return cluster
    return replace(
        cluster,
        remote_workspace=resolved_workspace,
        python_path=resolved_python,
    )


def runtime_activation_prefix(cluster: ClusterConfig) -> str:
    """Build a shell prefix for interactive remote commands."""
    parts = list(cluster.module_loads)
    if uses_managed_conda_env(cluster):
        parts.append(managed_conda_activate(cluster.remote_workspace, cluster.user))
    return " && ".join(parts)


def runtime_python_command(cluster: ClusterConfig) -> str:
    """Return a shell-ready Python executable for remote runtime commands."""
    if uses_managed_conda_env(cluster):
        return _MANAGED_ENV_PYTHON
    return shlex.quote(cluster.python_path.strip())


def runtime_setup_lines(cluster: ClusterConfig) -> list[str]:
    """Build shell lines needed before running on a compute node."""
    lines = list(cluster.module_loads)
    if uses_managed_conda_env(cluster):
        lines.append(managed_conda_bootstrap(cluster.remote_workspace, cluster.user))
        lines.append("conda activate crucible")
    lines.append("")
    return lines


def build_compute_node_command(
    cluster: ClusterConfig,
    resources: SlurmResourceConfig,
    inner_command: str,
) -> str:
    """Wrap a command in ``srun`` so it runs on a compute node."""
    partition = resources.partition or cluster.default_partition
    srun_parts = ["srun", "--nodes=1", "--ntasks=1"]
    if resources.gpus_per_node > 0:
        if resources.gpu_type:
            srun_parts.append(
                f"--gres=gpu:{resources.gpu_type}:{resources.gpus_per_node}"
            )
        else:
            srun_parts.append(f"--gres=gpu:{resources.gpus_per_node}")
    if partition:
        srun_parts.append(f"--partition={partition}")
    if cluster.exclude_nodes:
        srun_parts.append(f"--exclude={cluster.exclude_nodes}")
    if resources.memory:
        srun_parts.append(f"--mem={resources.memory}")
    if resources.time_limit:
        srun_parts.append(f"--time={resources.time_limit}")
    remote_parts = list(cluster.module_loads)
    if uses_managed_conda_env(cluster):
        remote_parts.append(managed_conda_bootstrap(cluster.remote_workspace, cluster.user))
        remote_parts.append("conda activate crucible")
    remote_parts.append(inner_command)
    srun_parts.append(f"bash -lc {shlex.quote(' && '.join(remote_parts))}")
    return " ".join(srun_parts)


def validation_resources(cluster: ClusterConfig) -> SlurmResourceConfig:
    """Return a lightweight GPU allocation for runtime validation."""
    return SlurmResourceConfig(
        partition=cluster.default_partition,
        nodes=1,
        gpus_per_node=1,
        gpu_type="",
        cpus_per_task=1,
        memory="8G",
        time_limit="00:03:00",
        extra_sbatch=(),
    )
