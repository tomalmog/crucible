"""Remote Slurm job log streaming and state synchronisation."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from core.errors import ForgeRemoteError
from core.slurm_types import TERMINAL_JOB_STATES, RemoteJobRecord, RemoteJobState
from serve.ssh_connection import SshSession
from store.cluster_registry import load_cluster
from store.remote_job_store import load_remote_job, update_remote_job_state


def stream_remote_logs(
    data_root: Path,
    job_id: str,
    tail_lines: int = 50,
) -> Generator[str, None, None]:
    """Stream logs from a running remote Slurm job.

    Opens an SSH connection and tails the Slurm output file in real time.
    Stops when the job reaches a terminal state.

    Args:
        data_root: Root .forge directory.
        job_id: Local job identifier.
        tail_lines: Number of existing lines to show initially.

    Yields:
        Individual log lines from the remote output file.
    """
    record = load_remote_job(data_root, job_id)
    if record.state in TERMINAL_JOB_STATES:
        yield from _fetch_static_logs(data_root, record, tail_lines)
        return

    cluster = load_cluster(data_root, record.cluster_name)
    log_path = record.remote_log_path
    if not log_path:
        log_path = f"{record.remote_output_dir}/slurm-{record.slurm_job_id}.out"

    with SshSession(cluster) as session:
        # Wait for log file to appear
        _, _, code = session.execute(
            f"test -f {log_path}", timeout=10,
        )
        if code != 0:
            yield f"[waiting for log file: {log_path}]"
            for _ in range(30):
                import time
                time.sleep(2)
                _, _, code = session.execute(
                    f"test -f {log_path}", timeout=10,
                )
                if code == 0:
                    break
            else:
                yield "[log file not found after 60s]"
                return

        for line in session.tail_follow(log_path, initial_lines=tail_lines):
            yield line
            # Periodically check if job is still running
            if _is_job_done(session, record.slurm_job_id):
                _sync_final_state(data_root, session, record)
                break


def fetch_remote_logs(
    data_root: Path,
    job_id: str,
    tail_lines: int = 100,
) -> str:
    """Fetch the last N lines of logs from a remote job."""
    record = load_remote_job(data_root, job_id)
    cluster = load_cluster(data_root, record.cluster_name)
    log_path = (
        record.remote_log_path
        or f"{record.remote_output_dir}/slurm-{record.slurm_job_id}.out"
    )

    with SshSession(cluster) as session:
        _, _, code = session.execute(f"test -f {log_path}", timeout=10)
        if code != 0:
            return (
                f"[Log file not found: {log_path}]\n"
                "The file may be on a compute node's local /tmp "
                "which is not accessible from the login node.\n"
                "Consider setting your cluster's Remote Workspace "
                "to a shared filesystem path (e.g. /home/you/forge-jobs)."
            )
        return session.tail_last(log_path, lines=tail_lines)


def check_remote_job_state(
    data_root: Path,
    job_id: str,
) -> RemoteJobState:
    """Check the current Slurm state of a remote job.

    Args:
        data_root: Root .forge directory.
        job_id: Local job identifier.

    Returns:
        Updated RemoteJobState.
    """
    record = load_remote_job(data_root, job_id)
    if record.state in TERMINAL_JOB_STATES:
        # Ensure model is registered even if state already terminal
        if (
            record.state == "completed"
            and record.model_path_remote
            and not _is_model_registered(data_root, record)
        ):
            _auto_register_remote_model(
                data_root, record, record.model_path_remote,
            )
        return record.state

    cluster = load_cluster(data_root, record.cluster_name)
    with SshSession(cluster) as session:
        return _sync_final_state(data_root, session, record)


def _is_job_done(session: SshSession, slurm_job_id: str) -> bool:
    """Check if a Slurm job has reached a terminal state."""
    stdout, _, code = session.execute(
        f"sacct -j {slurm_job_id} --noheader -o State -P", timeout=15,
    )
    if code != 0:
        return False
    states = [s.strip() for s in stdout.strip().splitlines() if s.strip()]
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"}
    return any(s in terminal for s in states)


def _slurm_state_to_forge(slurm_state: str) -> RemoteJobState:
    """Map a Slurm job state to a Forge RemoteJobState."""
    mapping: dict[str, RemoteJobState] = {
        "COMPLETED": "completed",
        "FAILED": "failed",
        "CANCELLED": "cancelled",
        "TIMEOUT": "failed",
        "NODE_FAIL": "failed",
        "PENDING": "pending",
        "RUNNING": "running",
    }
    return mapping.get(slurm_state.upper(), "running")


def _sync_final_state(
    data_root: Path,
    session: SshSession,
    record: RemoteJobRecord,
) -> RemoteJobState:
    """Query Slurm for the job's final state and persist it.

    When a job transitions to 'completed', reads result.json from the
    remote cluster to discover the model path and auto-registers it
    in the local model registry with location_type='remote'.
    """
    stdout, _, code = session.execute(
        f"sacct -j {record.slurm_job_id} --noheader -o State -P | head -1",
        timeout=15,
    )
    if code != 0:
        return record.state

    slurm_state = stdout.strip().splitlines()[0].strip() if stdout.strip() else ""
    forge_state = _slurm_state_to_forge(slurm_state)

    if forge_state != record.state:
        extra_fields: dict[str, str] = {}
        # On completion, discover model path and auto-register
        if forge_state == "completed" and not record.model_path_remote:
            model_path = _discover_remote_model(session, record)
            if model_path:
                extra_fields["model_path_remote"] = model_path
                _auto_register_remote_model(
                    data_root, record, model_path, session,
                )
        # On failure, extract error from Slurm log tail
        if forge_state == "failed":
            error_summary = _extract_log_error(session, record)
            if error_summary:
                extra_fields["submit_phase"] = f"Failed: {error_summary}"
        update_remote_job_state(
            data_root, record.job_id, forge_state, **extra_fields,
        )
    elif forge_state == "completed" and record.model_path_remote:
        # State already completed — ensure model is registered
        # (handles case where model_path was set but registration
        # failed or ran with older code)
        if not _is_model_registered(data_root, record):
            _auto_register_remote_model(
                data_root, record, record.model_path_remote, session,
            )
    return forge_state


def _extract_log_error(session: SshSession, record: RemoteJobRecord) -> str:
    """Try to extract an error message from the tail of the Slurm log."""
    log_path = (
        record.remote_log_path
        or f"{record.remote_output_dir}/slurm-{record.slurm_job_id}.out"
    )
    try:
        tail = session.tail_last(log_path, lines=30).strip()
    except Exception:
        return ""
    if not tail:
        return ""
    lines = tail.splitlines()
    error_patterns = ("Error:", "error:", "FAILED", "Traceback")
    for line in reversed(lines):
        if any(p in line for p in error_patterns):
            return line.strip()[:300]
    # No recognisable pattern — return the last non-empty line
    return next((l.strip()[:300] for l in reversed(lines) if l.strip()), "")


def _discover_remote_model(
    session: SshSession,
    record: RemoteJobRecord,
) -> str:
    """Read result.json from the remote to find the model path."""
    import json

    result_path = f"{record.remote_output_dir}/result.json"
    stdout, _, code = session.execute(f"cat '{result_path}'", timeout=15)
    if code != 0:
        return ""
    try:
        result = json.loads(stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return ""
    return str(result.get("model_path", ""))


def _auto_register_remote_model(
    data_root: Path,
    record: RemoteJobRecord,
    model_path: str,
    session: SshSession | None = None,
) -> None:
    """Register a completed remote model in the local model registry."""
    import logging

    try:
        from store.model_registry import ModelRegistry

        cluster = load_cluster(data_root, record.cluster_name)
        registry = ModelRegistry(data_root)
        model_name = (
            record.model_name
            or f"remote-{record.training_method}-{record.job_id[:16]}"
        )
        registry.register_remote_model(
            model_name=model_name,
            remote_host=cluster.host,
            remote_path=model_path,
            run_id=record.job_id,
        )
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to auto-register model for job %s", record.job_id,
            exc_info=True,
        )


def _is_model_registered(data_root: Path, record: RemoteJobRecord) -> bool:
    """Check if a remote job's model is already in the registry."""
    try:
        from store.model_registry import ModelRegistry

        registry = ModelRegistry(data_root)
        model_name = (
            record.model_name
            or f"remote-{record.training_method}-{record.job_id[:16]}"
        )
        versions = registry.list_versions_for_model(model_name)
        return any(v.run_id == record.job_id for v in versions)
    except Exception:
        return False


def _fetch_static_logs(
    data_root: Path,
    record: RemoteJobRecord,
    tail_lines: int,
) -> Generator[str, None, None]:
    """Yield lines from a completed job's log file."""
    try:
        content = fetch_remote_logs(data_root, record.job_id, tail_lines)
        yield from content.splitlines()
    except ForgeRemoteError:
        yield f"[could not retrieve logs for {record.job_id}]"
