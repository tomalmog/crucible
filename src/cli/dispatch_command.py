"""The ``crucible dispatch`` command — unified job submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from store.dataset_sdk import CrucibleClient


_MODEL_PATH_FIELDS = (
    "base_model", "base_model_path", "policy_model_path",
    "teacher_model_path", "student_model_path", "reference_model_path",
    "initial_weights_path",
)


def _resolve_model_names(
    data_root: Path, method_args: dict, remote: bool = False,
) -> dict:
    """Resolve Crucible model names to actual paths in method_args."""
    from store.model_registry import ModelRegistry
    try:
        registry = ModelRegistry(data_root)
    except Exception:
        return method_args

    for field in _MODEL_PATH_FIELDS:
        value = method_args.get(field)
        if not isinstance(value, str) or not value:
            continue
        if "/" in value or value.startswith(".") or value.endswith(".pt"):
            continue
        try:
            entry = registry.get_model(value)
            if remote and entry.remote_path:
                method_args[field] = entry.remote_path
            elif entry.model_path:
                method_args[field] = entry.model_path
        except Exception:
            pass
    return method_args


def _validate_model_fields(method_args: dict) -> None:
    """Raise early if any model-path field looks like a bare name.

    Catches common mistakes (passing a Crucible model name like 'jupiter'
    instead of a real path) before the job is submitted to a remote cluster.
    """
    for field in _MODEL_PATH_FIELDS:
        value = method_args.get(field)
        if not isinstance(value, str) or not value:
            continue
        if "/" in value or value.startswith(".") or value.endswith(".pt"):
            continue
        # Well-known single-word HF model IDs
        _KNOWN_SHORT_HF = {"gpt2", "distilgpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        if value in _KNOWN_SHORT_HF:
            continue
        raise ValueError(
            f"'{value}' in field '{field}' is not a valid model path or "
            f"HuggingFace model ID, and was not found in the model registry. "
            f"Use a full path (e.g. /path/to/model.pt) or a HuggingFace ID "
            f"(e.g. 'gpt2' or 'meta-llama/Llama-2-7b')."
        )


def add_dispatch_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "dispatch",
        help="Submit a job via JSON spec (used by Studio UI).",
    )
    parser.add_argument(
        "--spec", required=True,
        help="JSON-encoded JobSpec.",
    )


def run_dispatch_command(
    client: CrucibleClient, args: argparse.Namespace,
) -> int:
    """Parse spec JSON, look up backend, submit, print result."""
    from core.backend_registry import get_backend
    from core.job_types import JobSpec, ResourceConfig

    raw = json.loads(args.spec)

    resources = None
    if "resources" in raw and raw["resources"]:
        r = raw["resources"]
        resources = ResourceConfig(
            nodes=int(r.get("nodes", 1)),
            gpus_per_node=int(r.get("gpus_per_node", 1)),
            cpus_per_task=int(r.get("cpus_per_task", 4)),
            memory=str(r.get("memory", "32G")),
            time_limit=str(r.get("time_limit", "04:00:00")),
            partition=str(r.get("partition", "")),
            gpu_type=str(r.get("gpu_type", "")),
            extra_sbatch=tuple(
                tuple(pair) for pair in r.get("extra_sbatch", [])
            ),
        )

    sweep_trials = tuple(
        dict(t) for t in raw.get("sweep_trials", [])
    )

    method_args = dict(raw.get("method_args", {}))
    is_remote = str(raw.get("backend", "local")) != "local"
    method_args = _resolve_model_names(
        client._config.data_root, method_args, remote=is_remote,
    )
    _validate_model_fields(method_args)

    spec = JobSpec(
        job_type=str(raw["job_type"]),
        method_args=method_args,
        backend=str(raw.get("backend", "local")),  # type: ignore[arg-type]
        label=str(raw.get("label", "")),
        cluster_name=str(raw.get("cluster_name", "")),
        resources=resources,
        is_sweep=bool(raw.get("is_sweep", False)),
        sweep_trials=sweep_trials,
        config=dict(raw.get("config", {})),
    )

    backend = get_backend(spec.backend)
    try:
        record = backend.submit(client._config.data_root, spec)
    except Exception as exc:
        import traceback
        print(f"DISPATCH_ERROR: {exc}", flush=True)
        traceback.print_exc()
        # SSH and local runners write their own failed records; only
        # create a fallback for backends that don't manage job records.
        if spec.backend not in ("ssh", "local"):
            from store.job_store import generate_job_id, now_iso, save_job
            from core.job_types import JobRecord
            ts = now_iso()
            save_job(client._config.data_root, JobRecord(
                job_id=generate_job_id(),
                backend=spec.backend,
                job_type=spec.job_type,
                state="failed",
                created_at=ts,
                updated_at=ts,
                label=spec.label or spec.job_type,
                backend_cluster=spec.cluster_name,
                error_message=f"{type(exc).__name__}: {exc}",
            ))
        return 1

    result_dict = {
        "job_id": record.job_id,
        "backend": record.backend,
        "state": record.state,
        "job_type": record.job_type,
        "backend_job_id": record.backend_job_id,
        "model_path": record.model_path,
    }
    print(f"CRUCIBLE_JSON:{json.dumps(result_dict)}")
    return 0
