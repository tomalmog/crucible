"""Local execution backend — runs jobs in the current process."""

from __future__ import annotations

import traceback
from pathlib import Path

from core.errors import CrucibleError
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
        """Local results are parsed from stdout by the UI."""
        record = load_job(data_root, job_id)
        return {
            "job_id": record.job_id,
            "state": record.state,
            "model_path": record.model_path,
            "error_message": record.error_message,
        }
