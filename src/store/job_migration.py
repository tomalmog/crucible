"""Auto-migrate legacy remote-jobs to unified jobs store."""

from __future__ import annotations

from pathlib import Path

from core.job_types import JobRecord
from serve.training_run_io import read_json_file
from store.job_store import save_job, _jobs_dir


_MARKER_FILE = ".migrated_v1"


def _needs_migration(data_root: Path) -> bool:
    """Check if migration is needed (marker file absent)."""
    return not (_jobs_dir(data_root) / _MARKER_FILE).exists()


def migrate_remote_jobs(data_root: Path) -> int:
    """Migrate all .crucible/remote-jobs/*.json to .crucible/jobs/.

    Returns the number of records migrated.
    """
    remote_dir = data_root / "remote-jobs"
    if not remote_dir.exists():
        _write_marker(data_root)
        return 0

    count = 0
    for path in sorted(remote_dir.glob("rj-*.json")):
        raw = read_json_file(path)
        if not isinstance(raw, dict):
            continue
        record = _convert_remote_record(raw)
        save_job(data_root, record)
        count += 1

    _write_marker(data_root)
    return count


def ensure_migrated(data_root: Path) -> None:
    """Run migration if needed. Safe to call repeatedly."""
    if _needs_migration(data_root):
        migrate_remote_jobs(data_root)


def _write_marker(data_root: Path) -> None:
    """Write the migration marker file."""
    jobs = _jobs_dir(data_root)
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / _MARKER_FILE).write_text("migrated\n")


def _convert_remote_record(raw: dict[str, object]) -> JobRecord:
    """Convert a legacy RemoteJobRecord dict to a unified JobRecord."""
    # Use the legacy job_id as the unified job_id (keeps references stable)
    legacy_id = str(raw.get("job_id", ""))
    return JobRecord(
        job_id=legacy_id,
        backend="slurm",
        job_type=str(raw.get("training_method", "")),
        state=str(raw.get("state", "pending")),  # type: ignore[arg-type]
        created_at=str(raw.get("submitted_at", "")),
        updated_at=str(raw.get("updated_at", "")),
        label=str(raw.get("model_name", "")),
        backend_job_id=legacy_id,
        backend_cluster=str(raw.get("cluster_name", "")),
        backend_output_dir=str(raw.get("remote_output_dir", "")),
        backend_log_path=str(raw.get("remote_log_path", "")),
        model_path=str(raw.get("model_path_remote", "")),
        model_path_local=str(raw.get("model_path_local", "")),
        model_name=str(raw.get("model_name", "")),
        submit_phase=str(raw.get("submit_phase", "")),
        is_sweep=bool(raw.get("is_sweep", False)),
        sweep_trial_count=int(raw.get("sweep_array_size", 0)),  # type: ignore[arg-type]
    )
