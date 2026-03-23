"""Unified job types for all execution backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


JobState = Literal[
    "pending", "submitting", "running", "completed", "failed", "cancelled"
]

ALL_JOB_STATES: tuple[JobState, ...] = (
    "pending", "submitting", "running", "completed", "failed", "cancelled",
)

TERMINAL_JOB_STATES: frozenset[JobState] = frozenset({
    "completed", "failed", "cancelled",
})

BackendKind = Literal["local", "slurm", "docker-ssh", "http-api"]


@dataclass(frozen=True)
class ResourceConfig:
    """Resource allocation for remote backends."""

    nodes: int = 1
    gpus_per_node: int = 1
    cpus_per_task: int = 4
    memory: str = "32G"
    time_limit: str = "04:00:00"
    partition: str = ""
    gpu_type: str = ""
    extra_sbatch: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class JobSpec:
    """Backend-agnostic job specification. One format for all backends."""

    job_type: str
    method_args: dict[str, object] = field(default_factory=dict)
    backend: BackendKind = "local"
    label: str = ""
    cluster_name: str = ""
    resources: ResourceConfig | None = None
    is_sweep: bool = False
    sweep_trials: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class JobRecord:
    """Unified record for a job across all backends."""

    job_id: str
    backend: BackendKind
    job_type: str
    state: JobState
    created_at: str
    updated_at: str
    label: str = ""
    backend_job_id: str = ""
    backend_cluster: str = ""
    backend_output_dir: str = ""
    backend_log_path: str = ""
    model_path: str = ""
    model_path_local: str = ""
    model_name: str = ""
    error_message: str = ""
    progress_percent: float = 0.0
    submit_phase: str = ""
    is_sweep: bool = False
    sweep_trial_count: int = 0
