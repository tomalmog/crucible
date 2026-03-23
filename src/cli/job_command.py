"""The ``crucible job`` command — unified job management."""

from __future__ import annotations

import argparse
import json

from store.dataset_sdk import CrucibleClient


def add_job_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("job", help="Manage unified jobs.")
    sub = parser.add_subparsers(dest="job_action", required=True)

    sub.add_parser("list", help="List all jobs.")

    sync = sub.add_parser("sync", help="Sync a job's state from backend.")
    sync.add_argument("--job-id", required=True)

    cancel = sub.add_parser("cancel", help="Cancel a job.")
    cancel.add_argument("--job-id", required=True)

    logs = sub.add_parser("logs", help="Fetch job logs.")
    logs.add_argument("--job-id", required=True)
    logs.add_argument("--tail", type=int, default=200)

    result = sub.add_parser("result", help="Fetch job result.")
    result.add_argument("--job-id", required=True)


def run_job_command(
    client: CrucibleClient, args: argparse.Namespace,
) -> int:
    """Dispatch to the appropriate job sub-action."""
    data_root = client._config.data_root
    action = args.job_action

    if action == "list":
        return _handle_list(data_root)
    if action == "sync":
        return _handle_sync(data_root, args.job_id)
    if action == "cancel":
        return _handle_cancel(data_root, args.job_id)
    if action == "logs":
        return _handle_logs(data_root, args.job_id, args.tail)
    if action == "result":
        return _handle_result(data_root, args.job_id)
    return 1


def _handle_list(data_root: object) -> int:
    from pathlib import Path
    from store.job_store import list_jobs

    jobs = list_jobs(Path(str(data_root)))
    if not jobs:
        print("No jobs.")
        return 0
    for j in jobs:
        sweep_tag = f" (sweep, {j.sweep_trial_count} trials)" if j.is_sweep else ""
        print(
            f"  {j.job_id}  {j.job_type}  {j.state}  "
            f"backend={j.backend}{sweep_tag}"
        )
    return 0


def _handle_sync(data_root: object, job_id: str) -> int:
    from pathlib import Path
    from core.backend_registry import get_backend
    from store.job_store import load_job

    root = Path(str(data_root))
    record = load_job(root, job_id)
    backend = get_backend(record.backend)  # type: ignore[arg-type]
    new_state = backend.get_state(root, job_id)
    print(json.dumps({
        "job_id": record.job_id,
        "state": new_state,
        "backend": record.backend,
    }))
    return 0


def _handle_cancel(data_root: object, job_id: str) -> int:
    from pathlib import Path
    from core.backend_registry import get_backend
    from store.job_store import load_job

    root = Path(str(data_root))
    record = load_job(root, job_id)
    backend = get_backend(record.backend)  # type: ignore[arg-type]
    updated = backend.cancel(root, job_id)
    print(f"Cancelled job {updated.job_id}.")
    return 0


def _handle_logs(data_root: object, job_id: str, tail: int) -> int:
    from pathlib import Path
    from core.backend_registry import get_backend
    from store.job_store import load_job

    root = Path(str(data_root))
    record = load_job(root, job_id)
    backend = get_backend(record.backend)  # type: ignore[arg-type]
    content = backend.get_logs(root, job_id, tail=tail)
    print(content)
    return 0


def _handle_result(data_root: object, job_id: str) -> int:
    from pathlib import Path
    from core.backend_registry import get_backend
    from store.job_store import load_job

    root = Path(str(data_root))
    record = load_job(root, job_id)
    backend = get_backend(record.backend)  # type: ignore[arg-type]
    result = backend.get_result(root, job_id)
    print(f"CRUCIBLE_JSON:{json.dumps(result)}")
    return 0
