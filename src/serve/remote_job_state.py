"""Remote Slurm job state checking and synchronisation."""

from __future__ import annotations

from pathlib import Path

from core.slurm_types import TERMINAL_JOB_STATES, RemoteJobRecord, RemoteJobState
from serve.remote_model_registry import auto_register_remote_model, is_model_registered
from serve.remote_result_reader import extract_result_error, extract_result_model_path
from serve.ssh_connection import SshSession
from store.cluster_registry import load_cluster
from store.remote_job_store import load_remote_job, update_remote_job_state


def check_remote_job_state(
    data_root: Path,
    job_id: str,
) -> RemoteJobState:
    """Check the current Slurm state of a remote job.

    Args:
        data_root: Root .crucible directory.
        job_id: Local job identifier.

    Returns:
        Updated RemoteJobState.
    """
    record = load_remote_job(data_root, job_id)
    if record.state in TERMINAL_JOB_STATES:
        if record.state == "completed":
            _handle_completed_terminal(data_root, record)
        return record.state

    cluster = load_cluster(data_root, record.cluster_name)
    with SshSession(cluster) as session:
        return sync_final_state(data_root, session, record)


def _handle_completed_terminal(
    data_root: Path,
    record: RemoteJobRecord,
) -> None:
    """Handle model discovery/registration for completed terminal jobs."""
    if not record.model_path_remote:
        cluster = load_cluster(data_root, record.cluster_name)
        with SshSession(cluster) as session:
            model_path = extract_result_model_path(session, record)
            if model_path:
                update_remote_job_state(
                    data_root, record.job_id, "completed",
                    model_path_remote=model_path,
                )
                auto_register_remote_model(
                    data_root, record, model_path,
                )
    elif not is_model_registered(data_root, record):
        auto_register_remote_model(
            data_root, record, record.model_path_remote,
        )


def is_job_done(session: SshSession, slurm_job_id: str) -> bool:
    """Check if a Slurm job has reached a terminal state."""
    stdout, _, code = session.execute(
        f"sacct -j {slurm_job_id} --noheader -o State -P", timeout=15,
    )
    if code != 0:
        return False
    states = [s.strip() for s in stdout.strip().splitlines() if s.strip()]
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"}
    return any(s in terminal for s in states)


def slurm_state_to_crucible(slurm_state: str) -> RemoteJobState:
    """Map a Slurm job state to a Crucible RemoteJobState."""
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


def sync_final_state(
    data_root: Path,
    session: SshSession,
    record: RemoteJobRecord,
) -> RemoteJobState:
    """Query Slurm for the job's final state and persist it."""
    stdout, _, code = session.execute(
        f"sacct -j {record.slurm_job_id} --noheader "
        "-o State,ExitCode,Reason%50 -P | head -1",
        timeout=15,
    )
    if code != 0:
        return record.state

    parts = stdout.strip().splitlines()[0].split("|") if stdout.strip() else []
    slurm_state = parts[0].strip() if parts else ""
    slurm_exit = parts[1].strip() if len(parts) > 1 else ""
    slurm_reason = parts[2].strip() if len(parts) > 2 else ""
    crucible_state = slurm_state_to_crucible(slurm_state)

    if crucible_state != record.state:
        extra_fields = _build_state_transition_fields(
            session, record, crucible_state, data_root,
            slurm_state, slurm_exit, slurm_reason,
        )
        update_remote_job_state(
            data_root, record.job_id, crucible_state, **extra_fields,
        )
    elif crucible_state == "completed" and record.model_path_remote:
        if not is_model_registered(data_root, record):
            auto_register_remote_model(
                data_root, record, record.model_path_remote,
            )
    return crucible_state


def _build_state_transition_fields(
    session: SshSession,
    record: RemoteJobRecord,
    crucible_state: RemoteJobState,
    data_root: Path,
    slurm_state: str,
    slurm_exit: str,
    slurm_reason: str,
) -> dict[str, str]:
    """Build extra fields dict for a state transition update."""
    extra_fields: dict[str, str] = {}

    # Check result.json — agent may have failed even if Slurm says COMPLETED
    if crucible_state == "completed":
        result_error = extract_result_error(session, record)
        if result_error:
            crucible_state = "failed"
            extra_fields["submit_phase"] = f"Failed: {result_error}"

    # On completion, discover model path and auto-register
    if crucible_state == "completed" and not record.model_path_remote:
        import time
        model_path = ""
        for _attempt in range(8):
            model_path = extract_result_model_path(session, record)
            if model_path:
                break
            time.sleep(2)
        if model_path:
            extra_fields["model_path_remote"] = model_path
            auto_register_remote_model(data_root, record, model_path)

    # On failure, extract error from Slurm log tail or sacct
    if crucible_state == "failed":
        error_summary = extract_log_error(session, record)
        if not error_summary and slurm_reason and slurm_reason != "None":
            error_summary = (
                f"Slurm {slurm_state} (exit {slurm_exit}): {slurm_reason}"
            )
        elif not error_summary and slurm_exit:
            error_summary = f"Slurm {slurm_state} (exit {slurm_exit})"
        if error_summary:
            extra_fields["submit_phase"] = f"Failed: {error_summary}"

    return extra_fields


def query_sacct_details(session: SshSession, slurm_job_id: str) -> str:
    """Query sacct for detailed job information."""
    stdout, _, code = session.execute(
        f"sacct -j {slurm_job_id} --format=JobID,State,ExitCode,MaxRSS,"
        "Elapsed,NodeList,Reason%50 --noheader -P",
        timeout=15,
    )
    if code != 0 or not stdout.strip():
        return ""
    return stdout.strip()


def extract_log_error(session: SshSession, record: RemoteJobRecord) -> str:
    """Extract an error message from the remote job.

    Checks result.json first, then Slurm log tail, then sacct.
    """
    result_error = extract_result_error(session, record)
    if result_error:
        return result_error

    log_path = (
        record.remote_log_path
        or f"{record.remote_output_dir}/slurm-{record.slurm_job_id}.out"
    )
    error = _scan_log_tail(session, log_path)
    if error:
        return error

    return _extract_sacct_error(session, record.slurm_job_id)


def _scan_log_tail(session: SshSession, log_path: str) -> str:
    """Scan the tail of a log file for error patterns."""
    try:
        tail = session.tail_last(log_path, lines=50).strip()
    except Exception:
        return ""
    if not tail:
        return ""

    lines = tail.splitlines()
    error_patterns = (
        "CRUCIBLE_AGENT_ERROR:", "Error:", "error:", "Exception:",
        "FAILED", "Traceback", "Killed", "signal",
    )
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if any(p in stripped for p in error_patterns):
            return stripped[:300]

    skip_patterns = (
        "CUDA pre-flight OK", "CUDA_VISIBLE_DEVICES=", "+-", "|",
        "CRUCIBLE:", "Python ", "No running processes",
        "CRUCIBLE_AGENT:", "Mem:", "Swap:", "total",
    )
    for line in reversed(lines):
        stripped = line.strip()
        if stripped and not any(stripped.startswith(p) for p in skip_patterns):
            return stripped[:300]
    return ""


def _extract_sacct_error(session: SshSession, slurm_job_id: str) -> str:
    """Build a human-readable error from sacct output."""
    sacct_info = query_sacct_details(session, slurm_job_id)
    if not sacct_info:
        return ""
    for line in sacct_info.splitlines():
        parts = line.split("|")
        if len(parts) >= 3 and parts[0].strip() == slurm_job_id:
            state = parts[1].strip()
            exit_code = parts[2].strip()
            node = parts[5].strip() if len(parts) > 5 else ""
            reason = parts[6].strip() if len(parts) > 6 else ""
            msg = f"Slurm {state} (exit {exit_code})"
            if node:
                msg += f" on {node}"
            if reason and reason != "None":
                msg += f": {reason}"
            return msg[:300]
    return ""
