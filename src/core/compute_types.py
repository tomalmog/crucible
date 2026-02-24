"""Compute connectivity type definitions.

This module defines immutable data models for compute target management,
job submission and tracking, and connection profiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ComputeTarget:
    """Describes a compute backend to run jobs on.

    Attributes:
        name: Human-readable target identifier.
        executor_type: Backend type (only "local" supported now).
    """

    name: str
    executor_type: str = "local"


@dataclass(frozen=True)
class JobSubmission:
    """Encapsulates everything needed to launch a compute job.

    Attributes:
        target: The compute target to run on.
        command: Executable or script to invoke.
        args: Positional arguments passed to the command.
        working_dir: Optional working directory for execution.
        env_vars: Optional extra environment variables.
    """

    target: ComputeTarget
    command: str
    args: tuple[str, ...] = ()
    working_dir: str | None = None
    env_vars: dict[str, str] | None = None


@dataclass(frozen=True)
class JobStatus:
    """Tracks the lifecycle state of a submitted job.

    Attributes:
        job_id: Unique identifier for the job.
        state: Current state (queued, running, completed, failed).
        exit_code: Process exit code once finished.
        started_at: ISO timestamp when execution began.
        completed_at: ISO timestamp when execution ended.
    """

    job_id: str
    state: str
    exit_code: int | None = None
    started_at: str | None = None
    completed_at: str | None = None


@dataclass(frozen=True)
class ConnectionProfile:
    """Result of probing connectivity to a compute target.

    Attributes:
        target_name: Name of the target that was probed.
        connected: Whether the target is reachable.
        latency_ms: Round-trip latency in milliseconds.
    """

    target_name: str
    connected: bool
    latency_ms: float | None = None
