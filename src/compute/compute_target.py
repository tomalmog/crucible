"""Compute executor protocol and factory.

This module defines the abstract interface every compute backend must
satisfy and provides a factory to resolve executor instances by type.
"""

from __future__ import annotations

from typing import Protocol

from core.compute_types import JobStatus, JobSubmission
from core.errors import CrucibleComputeError


class ComputeExecutor(Protocol):
    """Protocol that all compute backends must implement."""

    def submit(self, submission: JobSubmission) -> str:
        """Submit a job and return its unique job_id."""
        ...

    def status(self, job_id: str) -> JobStatus:
        """Return the current status of a submitted job."""
        ...

    def cancel(self, job_id: str) -> bool:
        """Cancel a running job. Returns True if cancellation succeeded."""
        ...

    def fetch_artifacts(self, job_id: str, output_dir: str) -> list[str]:
        """Copy job artifacts to output_dir, return list of paths."""
        ...


def resolve_executor(executor_type: str) -> ComputeExecutor:
    """Resolve an executor instance by backend type string.

    Args:
        executor_type: Backend identifier (currently only "local").

    Returns:
        A concrete executor implementing ComputeExecutor.

    Raises:
        CrucibleComputeError: If the executor type is not supported.
    """
    if executor_type == "local":
        from compute.local_executor import LocalExecutor

        return LocalExecutor()
    raise CrucibleComputeError(
        f"Unsupported executor type: {executor_type!r}. "
        "Only 'local' is currently supported."
    )
