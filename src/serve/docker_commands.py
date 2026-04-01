"""Pure helper functions for building Docker CLI commands and parsing output."""

from __future__ import annotations

import shlex

from core.errors import CrucibleDockerError
from core.job_types import JobState


def build_gpu_flags(gpu_ids: str) -> str:
    """Return Docker --gpus flag for the given GPU device IDs.

    Args:
        gpu_ids: Comma-separated GPU IDs (e.g. "0,1") or empty for all.

    Returns:
        Docker GPU flag string.
    """
    if not gpu_ids.strip():
        return '--gpus all'
    return f'--gpus \'"device={gpu_ids.strip()}"\''


def build_docker_run_cmd(
    image: str,
    gpu_flags: str,
    volumes: tuple[tuple[str, str], ...],
    workdir: str,
    command: str,
) -> str:
    """Build a detached ``docker run`` command string.

    Args:
        image: Docker image name.
        gpu_flags: GPU flags from build_gpu_flags().
        volumes: Pairs of (host_path, container_path) for -v mounts.
        workdir: Working directory inside the container.
        command: Shell command to execute in the container.

    Returns:
        Full docker run command string.
    """
    parts = ["docker", "run", "-d", gpu_flags]
    for host_path, container_path in volumes:
        parts.append(f"-v {shlex.quote(host_path + ':' + container_path)}")
    parts.append(f"-w {shlex.quote(workdir)}")
    parts.append(shlex.quote(image))
    parts.append(f"bash -c {_shell_quote(command)}")
    return " ".join(parts)


def _shell_quote(text: str) -> str:
    """Wrap text in single quotes, escaping internal single quotes."""
    escaped = text.replace("'", "'\\''")
    return f"'{escaped}'"


def parse_container_id(stdout: str) -> str:
    """Extract a container ID from ``docker run -d`` output.

    Args:
        stdout: Raw stdout from the docker run command.

    Returns:
        The 12+ character container ID.

    Raises:
        CrucibleDockerError: If no valid container ID is found.
    """
    line = stdout.strip().split("\n")[-1].strip()
    if len(line) >= 12 and all(c in "0123456789abcdef" for c in line[:12]):
        return line[:12]
    raise CrucibleDockerError(
        f"Failed to parse container ID from docker output: {stdout[:200]}"
    )


_DOCKER_STATE_MAP: dict[str, JobState] = {
    "created": "pending",
    "running": "running",
    "paused": "running",
    "restarting": "running",
    "removing": "running",
    "exited": "completed",
    "dead": "failed",
}


def parse_docker_state(inspect_output: str) -> JobState:
    """Map a Docker container state string to a JobState.

    Args:
        inspect_output: Raw output from
            ``docker inspect --format='{{.State.Status}}'``.

    Returns:
        Corresponding JobState value.

    Raises:
        CrucibleDockerError: If the state is unrecognised.
    """
    state = inspect_output.strip().lower()
    if state in _DOCKER_STATE_MAP:
        return _DOCKER_STATE_MAP[state]
    raise CrucibleDockerError(
        f"Unrecognised Docker container state: '{state}'"
    )
