"""Local subprocess-based compute executor.

This module implements ComputeExecutor for the local machine,
launching jobs as child processes and tracking them by UUID.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any

from core.compute_types import JobStatus, JobSubmission
from core.errors import CrucibleComputeError


class LocalExecutor:
    """Execute compute jobs as local subprocesses."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    def submit(self, submission: JobSubmission) -> str:
        """Launch a subprocess for the given job submission.

        Args:
            submission: Job parameters including command and args.

        Returns:
            A unique job_id string.

        Raises:
            CrucibleComputeError: If the subprocess cannot be started.
        """
        job_id = uuid.uuid4().hex[:12]
        cmd = [submission.command, *submission.args]
        env = _build_env(submission.env_vars)
        try:
            process = subprocess.Popen(
                cmd,
                cwd=submission.working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise CrucibleComputeError(
                f"Failed to start process: {exc}"
            ) from exc
        self._jobs[job_id] = {
            "process": process,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "submission": submission,
        }
        return job_id

    def status(self, job_id: str) -> JobStatus:
        """Poll the subprocess and return current job status.

        Args:
            job_id: Identifier returned by submit().

        Returns:
            Current JobStatus snapshot.

        Raises:
            CrucibleComputeError: If job_id is unknown.
        """
        entry = self._get_job(job_id)
        process: subprocess.Popen[bytes] = entry["process"]
        return_code = process.poll()
        if return_code is None:
            return JobStatus(
                job_id=job_id,
                state="running",
                started_at=entry["started_at"],
            )
        state = "completed" if return_code == 0 else "failed"
        completed_at = datetime.now(timezone.utc).isoformat()
        return JobStatus(
            job_id=job_id,
            state=state,
            exit_code=return_code,
            started_at=entry["started_at"],
            completed_at=completed_at,
        )

    def cancel(self, job_id: str) -> bool:
        """Terminate a running subprocess.

        Args:
            job_id: Identifier returned by submit().

        Returns:
            True if the process was terminated, False if already done.

        Raises:
            CrucibleComputeError: If job_id is unknown.
        """
        entry = self._get_job(job_id)
        process: subprocess.Popen[bytes] = entry["process"]
        if process.poll() is not None:
            return False
        process.terminate()
        process.wait(timeout=5)
        return True

    def fetch_artifacts(self, job_id: str, output_dir: str) -> list[str]:
        """Return an empty artifact list (local jobs share filesystem).

        Args:
            job_id: Identifier returned by submit().
            output_dir: Destination directory (unused for local).

        Returns:
            Empty list since local jobs write directly to disk.
        """
        self._get_job(job_id)
        return []

    def _get_job(self, job_id: str) -> dict[str, Any]:
        """Look up a tracked job entry by id.

        Raises:
            CrucibleComputeError: If job_id is not found.
        """
        entry = self._jobs.get(job_id)
        if entry is None:
            raise CrucibleComputeError(f"Unknown job id: {job_id!r}")
        return entry


def _build_env(extra: dict[str, str] | None) -> dict[str, str]:
    """Merge extra env vars into the current environment.

    Args:
        extra: Additional environment variables to set.

    Returns:
        Combined environment dictionary.
    """
    env = dict(os.environ)
    if extra:
        env.update(extra)
    return env
