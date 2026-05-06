"""Local execution backend — runs jobs in the current process."""

from __future__ import annotations

import json
from pathlib import Path

from core.job_types import BackendKind, JobRecord, JobSpec, JobState
from core.training_methods import dispatch_training
from store.job_store import generate_job_id, now_iso, save_job, update_job, load_job


class LocalRunner:
    """Executes jobs as local subprocesses (via Rust task store) or in-process."""

    @property
    def kind(self) -> BackendKind:
        return "local"

    def submit(self, data_root: Path, spec: JobSpec) -> JobRecord:
        """Run the job in-process. Writes JobRecord before and after."""
        from store.dataset_sdk import CrucibleClient
        from core.config import CrucibleConfig

        job_id = generate_job_id()
        ts = now_iso()
        record = JobRecord(
            job_id=job_id,
            backend="local",
            job_type=spec.job_type,
            state="running",
            created_at=ts,
            updated_at=ts,
            label=spec.label,
            is_sweep=spec.is_sweep,
            config=spec.config,
        )
        save_job(data_root, record)

        try:
            config = CrucibleConfig.from_env()
            client = CrucibleClient(config)
            result = dispatch_training(client, spec.job_type, spec.method_args)
            model_path = str(result.model_path) if result.model_path else ""
            record = update_job(
                data_root, job_id,
                state="completed",
                model_path=model_path,
            )
            # Auto-register in model registry
            if model_path:
                from store.model_registry import ModelRegistry
                try:
                    model_name = spec.label or spec.job_type
                    registry = ModelRegistry(data_root)
                    registry.register_model(model_name, model_path, run_id=result.run_id)
                except Exception:
                    pass
            print(f"model_path={model_path}")
            print(f"job_id={job_id}")
        except Exception as exc:
            record = update_job(
                data_root, job_id,
                state="failed",
                error_message=str(exc),
            )
            raise
        return record

    def cancel(self, data_root: Path, job_id: str) -> JobRecord:
        """Cancel not meaningfully supported for in-process local jobs."""
        return update_job(data_root, job_id, state="cancelled")

    def get_state(self, data_root: Path, job_id: str) -> JobState:
        """Read current state from disk."""
        record = load_job(data_root, job_id)
        return record.state  # type: ignore[return-value]

    def get_logs(self, data_root: Path, job_id: str, tail: int = 200) -> str:
        """Local logs are streamed via Rust task store stdout, not stored here."""
        return ""

    def get_result(self, data_root: Path, job_id: str) -> dict[str, object]:
        """Read local result data from the persisted job record."""
        record = load_job(data_root, job_id)
        parsed = _parse_persisted_stdout(data_root, job_id)
        result: dict[str, object] = parsed if parsed is not None else {}
        result.update({
            "job_id": record.job_id,
            "state": record.state,
            "model_path": str(result.get("model_path") or record.model_path),
            "error_message": record.error_message,
        })
        return result


def _parse_persisted_stdout(data_root: Path, job_id: str) -> dict[str, object] | None:
    job_path = data_root / "jobs" / f"{job_id}.json"
    if not job_path.exists():
        return None
    try:
        raw = json.loads(job_path.read_text())
    except json.JSONDecodeError:
        return None
    stdout = raw.get("stdout")
    if not isinstance(stdout, str) or not stdout.strip():
        return None
    for line in reversed(stdout.splitlines()):
        payload = line.removeprefix("CRUCIBLE_JSON:")
        parsed = _try_parse_json_object(payload)
        if parsed is not None:
            return parsed
    return _try_parse_json_object(stdout)


def _try_parse_json_object(value: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
