"""CRUD operations for remote Slurm job records.

Persists RemoteJobRecord instances as JSON under .crucible/remote-jobs/.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.errors import CrucibleRemoteError
from core.slurm_types import RemoteJobRecord, RemoteJobState
from serve.training_run_io import read_json_file, write_json_file

_logger = logging.getLogger(__name__)


def _jobs_dir(data_root: Path) -> Path:
    """Return the remote-jobs storage directory, creating it if needed."""
    d = data_root / "remote-jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job_path(data_root: Path, job_id: str) -> Path:
    """Return the JSON file path for a remote job record."""
    return _jobs_dir(data_root) / f"{job_id}.json"


def generate_job_id() -> str:
    """Generate a unique remote job identifier."""
    return f"rj-{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _record_to_dict(record: RemoteJobRecord) -> dict[str, object]:
    """Serialize a RemoteJobRecord to a JSON-safe dictionary."""
    return {
        "job_id": record.job_id,
        "slurm_job_id": record.slurm_job_id,
        "cluster_name": record.cluster_name,
        "training_method": record.training_method,
        "state": record.state,
        "submitted_at": record.submitted_at,
        "updated_at": record.updated_at,
        "remote_output_dir": record.remote_output_dir,
        "remote_log_path": record.remote_log_path,
        "model_path_remote": record.model_path_remote,
        "model_path_local": record.model_path_local,
        "model_name": record.model_name,
        "is_sweep": record.is_sweep,
        "sweep_array_size": record.sweep_array_size,
        "submit_phase": record.submit_phase,
    }


def _dict_to_record(raw: dict[str, object]) -> RemoteJobRecord:
    """Reconstruct a RemoteJobRecord from a dictionary."""
    state_str = str(raw.get("state", "pending"))
    state: RemoteJobState = state_str if state_str in (  # type: ignore[assignment]
        "pending", "submitting", "running", "completed", "failed", "cancelled",
    ) else "pending"
    return RemoteJobRecord(
        job_id=str(raw["job_id"]),
        slurm_job_id=str(raw.get("slurm_job_id", "")),
        cluster_name=str(raw.get("cluster_name", "")),
        training_method=str(raw.get("training_method", "")),
        state=state,
        submitted_at=str(raw.get("submitted_at", "")),
        updated_at=str(raw.get("updated_at", "")),
        remote_output_dir=str(raw.get("remote_output_dir", "")),
        remote_log_path=str(raw.get("remote_log_path", "")),
        model_path_remote=str(raw.get("model_path_remote", "")),
        model_path_local=str(raw.get("model_path_local", "")),
        model_name=str(raw.get("model_name", "")),
        is_sweep=bool(raw.get("is_sweep", False)),
        sweep_array_size=int(raw.get("sweep_array_size", 0)),  # type: ignore[arg-type]
        submit_phase=str(raw.get("submit_phase", "")),
    )


def save_remote_job(data_root: Path, record: RemoteJobRecord) -> Path:
    """Persist a remote job record to .crucible/remote-jobs/.

    Args:
        data_root: Root .crucible directory.
        record: Remote job record to save.

    Returns:
        Path to the written JSON file.
    """
    target = _job_path(data_root, record.job_id)
    try:
        write_json_file(target, _record_to_dict(record))
    except Exception as error:
        raise CrucibleRemoteError(
            f"Failed to save remote job {record.job_id}: {error}."
        ) from error
    return target



def load_remote_job(data_root: Path, job_id: str) -> RemoteJobRecord:
    """Load a remote job record from disk.

    Args:
        data_root: Root .crucible directory.
        job_id: Job identifier to load.

    Returns:
        Loaded RemoteJobRecord instance.

    Raises:
        CrucibleRemoteError: If the job file does not exist or is invalid.
    """
    target = _job_path(data_root, job_id)
    if not target.exists():
        raise CrucibleRemoteError(f"Remote job '{job_id}' not found.")
    try:
        raw = read_json_file(target)
    except Exception as error:
        raise CrucibleRemoteError(
            f"Failed to load remote job {job_id}: {error}."
        ) from error
    if not isinstance(raw, dict):
        raise CrucibleRemoteError(f"Invalid remote job data for {job_id}.")
    return _dict_to_record(raw)


def list_remote_jobs(data_root: Path) -> tuple[RemoteJobRecord, ...]:
    """List all remote job records sorted by submission time (newest first).

    Args:
        data_root: Root .crucible directory.

    Returns:
        Tuple of RemoteJobRecord instances.
    """
    jobs_dir = _jobs_dir(data_root)
    records: list[RemoteJobRecord] = []
    for path in jobs_dir.glob("rj-*.json"):
        try:
            raw = read_json_file(path)
            if isinstance(raw, dict):
                records.append(_dict_to_record(raw))
        except Exception as exc:
            _logger.warning("Skipping corrupt remote job file '%s': %s", path.name, exc)
            continue
    records.sort(key=lambda r: r.submitted_at, reverse=True)
    return tuple(records)


def update_remote_job_state(
    data_root: Path,
    job_id: str,
    state: RemoteJobState,
    **extra: str,
) -> RemoteJobRecord:
    """Update the state of a remote job and persist to disk.

    Args:
        data_root: Root .crucible directory.
        job_id: Job identifier to update.
        state: New lifecycle state.
        **extra: Additional fields to update (model_path_remote, etc.).

    Returns:
        Updated RemoteJobRecord.
    """
    record = load_remote_job(data_root, job_id)
    updates: dict[str, object] = {"state": state, "updated_at": now_iso()}
    updates.update(extra)
    from dataclasses import replace
    updated = replace(record, **updates)  # type: ignore[arg-type]
    save_remote_job(data_root, updated)
    return updated
