"""Job management API endpoints for Crucible server.

Provides REST endpoints for submitting, listing, querying,
cancelling jobs and fetching logs/results.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from core.errors import CrucibleDependencyError


def _import_fastapi() -> Any:
    """Lazy-import fastapi, raising on absence."""
    try:
        import fastapi
        return fastapi
    except ImportError as exc:
        raise CrucibleDependencyError(
            "fastapi is required for the collaboration server. "
            "Install it with: pip install fastapi"
        ) from exc


def _check_token(headers: Any, fastapi: Any) -> None:
    """Verify the Authorization: Bearer token against CRUCIBLE_API_TOKEN."""
    expected = os.environ.get("CRUCIBLE_API_TOKEN", "")
    if not expected:
        return
    auth = headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise fastapi.HTTPException(status_code=401, detail="Missing token")
    if auth[len("Bearer "):] != expected:
        raise fastapi.HTTPException(status_code=403, detail="Invalid token")


def create_jobs_router(data_root_path: str) -> Any:
    """Create a FastAPI router for job management endpoints.

    Args:
        data_root_path: Path to the .crucible data root directory.

    Returns:
        Configured FastAPI APIRouter.
    """
    fastapi = _import_fastapi()
    router = fastapi.APIRouter(prefix="/api/jobs", tags=["jobs"])
    data_root = Path(data_root_path).expanduser().resolve()

    @router.post("", status_code=201)
    def submit_job(
        request: Any,
        body: dict[str, Any] = fastapi.Body(...),
    ) -> dict[str, Any]:
        """Submit a new job via JSON spec."""
        _check_token(request.headers, fastapi)
        from core.backend_registry import get_backend
        from core.job_types import JobSpec, ResourceConfig

        resources = _parse_resources(body.get("resources"))
        spec = JobSpec(
            job_type=str(body["job_type"]),
            method_args=dict(body.get("method_args", {})),
            backend=str(body.get("backend", "local")),  # type: ignore[arg-type]
            label=str(body.get("label", "")),
            cluster_name=str(body.get("cluster_name", "")),
            resources=resources,
            config=dict(body.get("config", {})),
        )
        backend = get_backend(spec.backend)
        record = backend.submit(data_root, spec)
        return _serialize_record(record)

    @router.get("")
    def list_jobs(request: Any) -> list[dict[str, Any]]:
        """List all jobs."""
        _check_token(request.headers, fastapi)
        from store.job_store import list_jobs as store_list_jobs

        records = store_list_jobs(data_root)
        return [_serialize_record(r) for r in records]

    @router.get("/{job_id}")
    def get_job(job_id: str, request: Any) -> dict[str, Any]:
        """Get a single job by ID."""
        _check_token(request.headers, fastapi)
        from store.job_store import load_job

        record = load_job(data_root, job_id)
        return _serialize_record(record)

    @router.get("/{job_id}/logs")
    def get_job_logs(
        job_id: str, request: Any, tail: int = 200,
    ) -> dict[str, str]:
        """Get logs for a job."""
        _check_token(request.headers, fastapi)
        from core.backend_registry import get_backend
        from store.job_store import load_job

        record = load_job(data_root, job_id)
        backend = get_backend(record.backend)  # type: ignore[arg-type]
        logs = backend.get_logs(data_root, job_id, tail=tail)
        return {"logs": logs}

    @router.get("/{job_id}/result")
    def get_job_result(job_id: str, request: Any) -> dict[str, Any]:
        """Get the result of a completed job."""
        _check_token(request.headers, fastapi)
        from core.backend_registry import get_backend
        from store.job_store import load_job

        record = load_job(data_root, job_id)
        backend = get_backend(record.backend)  # type: ignore[arg-type]
        return backend.get_result(data_root, job_id)

    @router.post("/{job_id}/cancel")
    def cancel_job(job_id: str, request: Any) -> dict[str, Any]:
        """Cancel a running job."""
        _check_token(request.headers, fastapi)
        from core.backend_registry import get_backend
        from store.job_store import load_job

        record = load_job(data_root, job_id)
        backend = get_backend(record.backend)  # type: ignore[arg-type]
        updated = backend.cancel(data_root, job_id)
        return _serialize_record(updated)

    return router


def _parse_resources(raw: dict[str, Any] | None) -> Any:
    """Parse a resource config dict into a ResourceConfig."""
    if not raw:
        return None
    from core.job_types import ResourceConfig

    return ResourceConfig(
        nodes=int(raw.get("nodes", 1)),
        gpus_per_node=int(raw.get("gpus_per_node", 1)),
        cpus_per_task=int(raw.get("cpus_per_task", 4)),
        memory=str(raw.get("memory", "32G")),
        time_limit=str(raw.get("time_limit", "04:00:00")),
        partition=str(raw.get("partition", "")),
        gpu_type=str(raw.get("gpu_type", "")),
    )


def _serialize_record(record: Any) -> dict[str, Any]:
    """Convert a JobRecord to a JSON-safe dictionary."""
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
        "model_path": record.model_path,
        "error_message": record.error_message,
        "progress_percent": record.progress_percent,
    }
