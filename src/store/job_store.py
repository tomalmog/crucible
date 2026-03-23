"""CRUD store for unified job records in .crucible/jobs/."""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from core.errors import CrucibleError
from core.job_types import JobRecord, JobState
from serve.training_run_io import read_json_file, write_json_file


def _jobs_dir(data_root: Path) -> Path:
    return data_root / "jobs"


def _job_path(data_root: Path, job_id: str) -> Path:
    return _jobs_dir(data_root) / f"{job_id}.json"


def generate_job_id() -> str:
    """Generate a unique job identifier."""
    return f"job-{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    """UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _record_to_dict(record: JobRecord) -> dict[str, object]:
    return {
        "job_id": record.job_id,
        "backend": record.backend,
        "job_type": record.job_type,
        "state": record.state,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "label": record.label,
        "backend_job_id": record.backend_job_id,
        "backend_cluster": record.backend_cluster,
        "backend_output_dir": record.backend_output_dir,
        "backend_log_path": record.backend_log_path,
        "model_path": record.model_path,
        "model_path_local": record.model_path_local,
        "model_name": record.model_name,
        "error_message": record.error_message,
        "progress_percent": record.progress_percent,
        "submit_phase": record.submit_phase,
        "is_sweep": record.is_sweep,
        "sweep_trial_count": record.sweep_trial_count,
    }


def _dict_to_record(raw: dict[str, object]) -> JobRecord:
    return JobRecord(
        job_id=str(raw.get("job_id", "")),
        backend=str(raw.get("backend", "local")),  # type: ignore[arg-type]
        job_type=str(raw.get("job_type", "")),
        state=str(raw.get("state", "pending")),  # type: ignore[arg-type]
        created_at=str(raw.get("created_at", "")),
        updated_at=str(raw.get("updated_at", "")),
        label=str(raw.get("label", "")),
        backend_job_id=str(raw.get("backend_job_id", "")),
        backend_cluster=str(raw.get("backend_cluster", "")),
        backend_output_dir=str(raw.get("backend_output_dir", "")),
        backend_log_path=str(raw.get("backend_log_path", "")),
        model_path=str(raw.get("model_path", "")),
        model_path_local=str(raw.get("model_path_local", "")),
        model_name=str(raw.get("model_name", "")),
        error_message=str(raw.get("error_message", "")),
        progress_percent=float(raw.get("progress_percent", 0.0)),  # type: ignore[arg-type]
        submit_phase=str(raw.get("submit_phase", "")),
        is_sweep=bool(raw.get("is_sweep", False)),
        sweep_trial_count=int(raw.get("sweep_trial_count", 0)),  # type: ignore[arg-type]
    )


def save_job(data_root: Path, record: JobRecord) -> Path:
    """Write a job record to disk. Returns the file path."""
    jobs = _jobs_dir(data_root)
    jobs.mkdir(parents=True, exist_ok=True)
    path = _job_path(data_root, record.job_id)
    write_json_file(path, _record_to_dict(record))
    return path


def load_job(data_root: Path, job_id: str) -> JobRecord:
    """Load a single job record. Raises CrucibleError if not found."""
    path = _job_path(data_root, job_id)
    if not path.exists():
        raise CrucibleError(f"Job '{job_id}' not found")
    raw = read_json_file(path)
    if not isinstance(raw, dict):
        raise CrucibleError(f"Invalid job file at {path}")
    return _dict_to_record(raw)


def list_jobs(data_root: Path) -> tuple[JobRecord, ...]:
    """List all job records, newest first. Auto-migrates legacy records."""
    from store.job_migration import ensure_migrated
    ensure_migrated(data_root)
    jobs = _jobs_dir(data_root)
    if not jobs.exists():
        return ()
    records: list[JobRecord] = []
    _VALID_PREFIXES = ("job-", "crucible-task-", "rj-")
    for path in sorted(jobs.glob("*.json"), reverse=True):
        name = path.stem
        if not any(name.startswith(p) for p in _VALID_PREFIXES):
            continue
        raw = read_json_file(path)
        if isinstance(raw, dict):
            records.append(_dict_to_record(raw))
    records.sort(key=lambda r: r.created_at, reverse=True)
    return tuple(records)


def update_job(data_root: Path, job_id: str, **fields: object) -> JobRecord:
    """Update specific fields on a job record. Returns updated record."""
    record = load_job(data_root, job_id)
    fields["updated_at"] = now_iso()
    updated = replace(record, **fields)  # type: ignore[arg-type]
    save_job(data_root, updated)
    return updated


def delete_job(data_root: Path, job_id: str) -> None:
    """Delete a job record file."""
    path = _job_path(data_root, job_id)
    if not path.exists():
        raise CrucibleError(f"Job '{job_id}' not found")
    path.unlink()
