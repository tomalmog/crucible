"""State-checking helpers for the SSH execution backend.

Inspects Docker container state or bare SSH process state via SSH.
Handles model registration on job completion.
Used by SshRunner.get_state().
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from core.errors import CrucibleDockerError
from core.job_types import JobRecord, JobState

_log = logging.getLogger(__name__)


def get_docker_state(session: object, record: JobRecord) -> JobState:
    """Inspect Docker container state and exit code."""
    from serve.docker_commands import parse_docker_state
    from serve.ssh_connection import SshSession
    if not isinstance(session, SshSession):
        return "failed"
    cid = record.backend_job_id
    fmt = "docker inspect --format='{{{{.State.{field}}}}}' {cid}"
    stdout, stderr, code = session.execute(
        fmt.format(field="Status", cid=cid), timeout=15,
    )
    if code != 0:
        raise CrucibleDockerError(f"Docker inspect failed: {stderr.strip()}")
    state = parse_docker_state(stdout)
    if state != "completed":
        return state
    stdout, _, code = session.execute(
        fmt.format(field="ExitCode", cid=cid), timeout=15,
    )
    if code != 0:
        return "completed"
    try:
        return "completed" if int(stdout.strip()) == 0 else "failed"
    except ValueError:
        return "completed"


def get_bare_state(session: object, record: JobRecord) -> JobState:
    """Check bare SSH process state via log markers, result.json, and kill -0.

    Priority order (each is checked before moving to the next):
    1. Log markers — authoritative signal written by the agent.
    2. result.json status — written by the agent on completion/failure.
    3. kill -0 PID check — only trusted when (1) and (2) are absent,
       since the PID from ``nohup`` may have been reused.
    """
    from serve.ssh_connection import SshSession
    if not isinstance(session, SshSession):
        return "failed"

    # 1. Check log markers — authoritative signal of completion.
    log_path = session.resolve_path(
        f"{record.backend_output_dir}/output/train.log",
    )
    stdout, _, _ = session.execute(
        f"tail -50 {log_path} 2>/dev/null", timeout=10,
    )
    if "CRUCIBLE_AGENT_COMPLETE" in stdout:
        return "completed"
    if "CRUCIBLE_AGENT_ERROR" in stdout:
        return "failed"

    # 2. Check result.json — the agent writes this on success or failure.
    #    Checked before kill -0 so PID reuse can't override a definitive
    #    completed/failed result.
    result_path = session.resolve_path(
        f"{record.backend_output_dir}/output/result.json",
    )
    rout, _, rcode = session.execute(
        f"cat {result_path} 2>/dev/null", timeout=10,
    )
    if rcode == 0 and rout.strip():
        try:
            result = json.loads(rout.strip())
            status = result.get("status", "")
            if status == "completed":
                return "completed"
            if status == "failed":
                return "failed"
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. No markers and no result.json — check if the process is alive.
    pid = record.backend_job_id
    if pid:
        _, _, code = session.execute(
            f"kill -0 {pid} 2>/dev/null", timeout=10,
        )
        if code == 0:
            # Guard against PID reuse: if log file hasn't been modified
            # in 30+ minutes, the original process likely exited and the
            # PID was reassigned to an unrelated process.
            mtime_out, _, mtime_code = session.execute(
                f"stat -c %Y {log_path} 2>/dev/null || stat -f %m {log_path} 2>/dev/null",
                timeout=10,
            )
            if mtime_code == 0 and mtime_out.strip():
                import time
                try:
                    mtime = int(mtime_out.strip())
                    if time.time() - mtime > 1800:
                        _log.warning(
                            "Job %s PID %s alive but log stale (>30min) — likely PID reuse",
                            record.job_id, pid,
                        )
                        return "failed"
                except ValueError:
                    pass
            return "running"

    return "failed"


def handle_ssh_completion(
    session: object, data_root: Path, record: JobRecord,
) -> None:
    """Extract model path from result.json and register in model registry.

    Called when an SSH job transitions to 'completed'.  Skips eval jobs
    (they don't produce models) and jobs that already have a model_path.
    """
    from serve.ssh_connection import SshSession
    from store.job_store import update_job

    if not isinstance(session, SshSession):
        return
    if record.model_path:
        return
    if record.job_type == "eval":
        return

    result_path = session.resolve_path(
        f"{record.backend_output_dir}/output/result.json",
    )
    stdout, _, code = session.execute(f"cat {result_path}", timeout=15)
    if code != 0:
        return
    try:
        result = json.loads(stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return

    model_path = str(result.get("model_path", ""))
    if not model_path:
        return

    update_job(data_root, record.job_id, model_path=model_path)
    _register_model(data_root, record, model_path)


def _register_model(
    data_root: Path, record: JobRecord, model_path: str,
) -> None:
    """Register a completed SSH job's model in the model registry."""
    try:
        from store.cluster_registry import load_cluster
        from store.model_registry import ModelRegistry

        cluster = load_cluster(data_root, record.backend_cluster)
        registry = ModelRegistry(data_root)
        model_name = (
            record.model_name
            or f"remote-{record.job_type}-{record.job_id[:16]}"
        )
        registry.register_remote_model(
            model_name=model_name,
            remote_host=cluster.host,
            remote_path=model_path,
            run_id=record.job_id,
        )
    except Exception:
        _log.warning(
            "Failed to auto-register model for job %s", record.job_id,
            exc_info=True,
        )
