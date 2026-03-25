"""Typed models for remote Slurm cluster management.

This module defines immutable data models for cluster configuration,
resource allocation, and remote job tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RemoteJobState = Literal[
    "pending", "submitting", "running", "completed", "failed", "cancelled"
]

ALL_REMOTE_JOB_STATES: tuple[RemoteJobState, ...] = (
    "pending", "submitting", "running", "completed", "failed", "cancelled",
)

TERMINAL_JOB_STATES: frozenset[RemoteJobState] = frozenset({
    "completed", "failed", "cancelled",
})


@dataclass(frozen=True)
class ClusterConfig:
    """Registered Slurm cluster connection details.

    Attributes:
        name: Unique cluster identifier.
        host: SSH hostname or ~/.ssh/config alias.
        user: SSH username.
        ssh_key_path: Path to private key (empty = ssh agent/config).
        password: SSH password (empty = key-based or agent auth).
        default_partition: Default Slurm partition to submit to.
        partitions: Available Slurm partitions.
        gpu_types: Available GPU types on this cluster.
        module_loads: Shell commands to load required modules.
        python_path: Path to python interpreter on the remote.
        remote_workspace: Base directory for job files on the remote.
        exclude_nodes: Comma-separated node names to exclude from scheduling.
        validated_at: ISO-8601 timestamp of last successful validation.
    """

    name: str
    host: str
    user: str
    ssh_port: int = 22
    ssh_key_path: str = ""
    password: str = ""
    default_partition: str = ""
    partitions: tuple[str, ...] = ()
    gpu_types: tuple[str, ...] = ()
    module_loads: tuple[str, ...] = ()
    python_path: str = "python3"
    remote_workspace: str = "~/crucible-jobs"
    exclude_nodes: str = ""
    validated_at: str = ""
    backend: str = "slurm"
    docker_image: str = ""
    api_endpoint: str = ""
    api_token: str = ""


@dataclass(frozen=True)
class SlurmResourceConfig:
    """Resource allocation for a Slurm job submission.

    Attributes:
        partition: Slurm partition name.
        nodes: Number of nodes to allocate.
        gpus_per_node: GPUs requested per node.
        gpu_type: GPU type constraint (e.g. "a100").
        cpus_per_task: CPU cores per task.
        memory: Memory limit string (e.g. "32G").
        time_limit: Wall-clock limit (e.g. "12:00:00").
        extra_sbatch: Additional sbatch key=value pairs.
    """

    partition: str = ""
    nodes: int = 1
    gpus_per_node: int = 1
    gpu_type: str = ""
    cpus_per_task: int = 4
    memory: str = "32G"
    time_limit: str = "12:00:00"
    extra_sbatch: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class RemoteJobRecord:
    """Persistent record of a submitted remote Slurm job.

    Attributes:
        job_id: Local identifier ("rj-{uuid12}").
        slurm_job_id: Slurm job ID from sbatch output.
        cluster_name: Name of the target cluster.
        training_method: Training method used (e.g. "sft").
        state: Current lifecycle state.
        submitted_at: ISO-8601 submission timestamp.
        updated_at: ISO-8601 last state-change timestamp.
        remote_output_dir: Working directory on the remote.
        remote_log_path: Path to Slurm log file on remote.
        model_path_remote: Path to trained model on remote.
        model_path_local: Local path after pull (empty if not pulled).
        is_sweep: Whether this is a sweep job array.
        sweep_array_size: Number of trials if sweep.
    """

    job_id: str
    slurm_job_id: str
    cluster_name: str
    training_method: str
    state: RemoteJobState
    submitted_at: str
    updated_at: str
    remote_output_dir: str
    remote_log_path: str = ""
    model_path_remote: str = ""
    model_path_local: str = ""
    model_name: str = ""
    is_sweep: bool = False
    sweep_array_size: int = 0
    submit_phase: str = ""


@dataclass(frozen=True)
class ClusterValidationResult:
    """Result of validating a remote cluster's readiness.

    Attributes:
        cluster_name: Name of the validated cluster.
        python_ok: Whether Python is accessible.
        python_version: Detected Python version string.
        torch_ok: Whether PyTorch is importable.
        torch_version: Detected PyTorch version string.
        cuda_ok: Whether CUDA is available via torch.
        cuda_version: Detected CUDA version string.
        slurm_ok: Whether sinfo/sbatch are available.
        docker_ok: Whether Docker is available.
        docker_gpu_ok: Whether Docker has GPU access.
        partitions: Discovered Slurm partitions.
        gpu_types: Discovered GPU types.
        module_suggestions: Suggested module load commands.
        errors: List of validation error messages.
    """

    cluster_name: str
    python_ok: bool = False
    python_version: str = ""
    torch_ok: bool = False
    torch_version: str = ""
    cuda_ok: bool = False
    cuda_version: str = ""
    slurm_ok: bool = False
    docker_ok: bool = False
    docker_gpu_ok: bool = False
    partitions: tuple[str, ...] = ()
    gpu_types: tuple[str, ...] = ()
    module_suggestions: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
