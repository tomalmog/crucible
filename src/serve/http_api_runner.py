"""HTTP API execution backend — submits jobs to a remote Crucible server."""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from pathlib import Path

from core.errors import CrucibleRemoteError
from core.job_types import BackendKind, JobRecord, JobSpec, JobState
from store.job_store import generate_job_id, now_iso, save_job, load_job, update_job


class HttpApiRunner:
    """Delegates job execution to a remote Crucible server via HTTP API."""

    @property
    def kind(self) -> BackendKind:
        return "http-api"

    def submit(self, data_root: Path, spec: JobSpec) -> JobRecord:
        """Submit a job to the remote server."""
        from store.cluster_registry import load_cluster

        cluster = load_cluster(data_root, spec.cluster_name)
        endpoint = cluster.api_endpoint
        token = cluster.api_token

        body = _build_submit_body(spec)
        response = _api_post(endpoint, "/api/jobs", token, body)

        remote_job_id = str(response.get("job_id", ""))
        remote_state = str(response.get("state", "pending"))

        ts = now_iso()
        record = JobRecord(
            job_id=generate_job_id(),
            backend="http-api",
            job_type=spec.job_type,
            state=remote_state,  # type: ignore[arg-type]
            created_at=ts,
            updated_at=ts,
            label=spec.label or spec.job_type,
            backend_job_id=remote_job_id,
            backend_cluster=spec.cluster_name,
            config=spec.config or {},
        )
        save_job(data_root, record)
        return record

    def cancel(self, data_root: Path, job_id: str) -> JobRecord:
        """Cancel a job on the remote server."""
        from store.cluster_registry import load_cluster

        record = load_job(data_root, job_id)
        cluster = load_cluster(data_root, record.backend_cluster)

        _api_post(
            cluster.api_endpoint,
            f"/api/jobs/{record.backend_job_id}/cancel",
            cluster.api_token,
            {},
        )
        return update_job(data_root, job_id, state="cancelled")

    def get_state(self, data_root: Path, job_id: str) -> JobState:
        """Query job state from the remote server."""
        from store.cluster_registry import load_cluster

        record = load_job(data_root, job_id)
        cluster = load_cluster(data_root, record.backend_cluster)

        response = _api_get(
            cluster.api_endpoint,
            f"/api/jobs/{record.backend_job_id}",
            cluster.api_token,
        )
        state = str(response.get("state", "pending"))
        update_job(data_root, job_id, state=state)
        return state  # type: ignore[return-value]

    def get_logs(
        self, data_root: Path, job_id: str, tail: int = 200,
    ) -> str:
        """Fetch job logs from the remote server."""
        from store.cluster_registry import load_cluster

        record = load_job(data_root, job_id)
        cluster = load_cluster(data_root, record.backend_cluster)

        response = _api_get(
            cluster.api_endpoint,
            f"/api/jobs/{record.backend_job_id}/logs?tail={tail}",
            cluster.api_token,
        )
        return str(response.get("logs", ""))

    def get_result(self, data_root: Path, job_id: str) -> dict[str, object]:
        """Fetch job result from the remote server."""
        from store.cluster_registry import load_cluster

        record = load_job(data_root, job_id)
        cluster = load_cluster(data_root, record.backend_cluster)

        return _api_get(
            cluster.api_endpoint,
            f"/api/jobs/{record.backend_job_id}/result",
            cluster.api_token,
        )


def _build_submit_body(spec: JobSpec) -> dict[str, object]:
    """Build the JSON body for POST /api/jobs."""
    body: dict[str, object] = {
        "job_type": spec.job_type,
        "method_args": dict(spec.method_args),
        "backend": spec.backend,
        "label": spec.label,
        "cluster_name": spec.cluster_name,
    }
    if spec.resources:
        body["resources"] = {
            "nodes": spec.resources.nodes,
            "gpus_per_node": spec.resources.gpus_per_node,
            "cpus_per_task": spec.resources.cpus_per_task,
            "memory": spec.resources.memory,
            "time_limit": spec.resources.time_limit,
            "partition": spec.resources.partition,
            "gpu_type": spec.resources.gpu_type,
        }
    if spec.config:
        body["config"] = dict(spec.config)
    return body


def _api_get(
    endpoint: str, path: str, token: str,
) -> dict[str, object]:
    """Make an authenticated GET request to the API."""
    url = endpoint.rstrip("/") + path
    request = urllib.request.Request(url, method="GET")
    _set_auth_header(request, token)
    return _execute_request(request)


def _api_post(
    endpoint: str, path: str, token: str, body: dict[str, object],
) -> dict[str, object]:
    """Make an authenticated POST request to the API."""
    url = endpoint.rstrip("/") + path
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    _set_auth_header(request, token)
    return _execute_request(request)


def _set_auth_header(request: urllib.request.Request, token: str) -> None:
    """Set the Authorization header if a token is provided."""
    if token:
        request.add_header("Authorization", f"Bearer {token}")


def _execute_request(request: urllib.request.Request) -> dict[str, object]:
    """Execute an HTTP request and parse the JSON response."""
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)  # type: ignore[no-any-return]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise CrucibleRemoteError(
            f"HTTP API request failed ({exc.code}): {body[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise CrucibleRemoteError(
            f"HTTP API connection failed: {exc.reason}"
        ) from exc
