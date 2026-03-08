"""Remote Slurm job log streaming and fetching."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from core.errors import ForgeRemoteError
from core.slurm_types import TERMINAL_JOB_STATES
from serve.remote_job_state import is_job_done, query_sacct_details, sync_final_state
from serve.ssh_connection import SshSession
from store.cluster_registry import load_cluster
from store.remote_job_store import load_remote_job


def stream_remote_logs(
    data_root: Path,
    job_id: str,
    tail_lines: int = 50,
) -> Generator[str, None, None]:
    """Stream logs from a running remote Slurm job.

    Opens an SSH connection and tails the Slurm output file in real time.
    Stops when the job reaches a terminal state.

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
        yield from _wait_for_log_file(session, log_path)
        if not _log_file_exists(session, log_path):
            return

        for line in session.tail_follow(log_path, initial_lines=tail_lines):
            yield line
            if is_job_done(session, record.slurm_job_id):
                sync_final_state(data_root, session, record)
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
            sacct_info = query_sacct_details(session, record.slurm_job_id)
            return (
                f"[Log file not found: {log_path}]\n"
                "The file may be on a compute node's local /tmp "
                "which is not accessible from the login node.\n"
                "Consider setting your cluster's Remote Workspace "
                "to a shared filesystem path (e.g. /home/you/forge-jobs)."
                + (f"\n\n--- Slurm Job Info (sacct) ---\n{sacct_info}" if sacct_info else "")
            )
        return session.tail_last(log_path, lines=tail_lines)


def _wait_for_log_file(
    session: SshSession,
    log_path: str,
) -> Generator[str, None, None]:
    """Wait for a log file to appear, yielding status messages."""
    _, _, code = session.execute(f"test -f {log_path}", timeout=10)
    if code == 0:
        return

    yield f"[waiting for log file: {log_path}]"
    import time
    for _ in range(30):
        time.sleep(2)
        _, _, code = session.execute(f"test -f {log_path}", timeout=10)
        if code == 0:
            return
    yield "[log file not found after 60s]"


def _log_file_exists(session: SshSession, log_path: str) -> bool:
    """Check if a log file exists on the remote."""
    _, _, code = session.execute(f"test -f {log_path}", timeout=10)
    return code == 0


def _fetch_static_logs(
    data_root: Path,
    record: object,
    tail_lines: int,
) -> Generator[str, None, None]:
    """Yield lines from a completed job's log file."""
    try:
        content = fetch_remote_logs(data_root, record.job_id, tail_lines)  # type: ignore[union-attr]
        yield from content.splitlines()
    except ForgeRemoteError:
        yield f"[could not retrieve logs for {record.job_id}]"  # type: ignore[union-attr]
