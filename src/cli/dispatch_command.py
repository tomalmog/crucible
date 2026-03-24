"""The ``crucible dispatch`` command — unified job submission."""

from __future__ import annotations

import argparse
import json

from store.dataset_sdk import CrucibleClient


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

    spec = JobSpec(
        job_type=str(raw["job_type"]),
        method_args=dict(raw.get("method_args", {})),
        backend=str(raw.get("backend", "local")),  # type: ignore[arg-type]
        label=str(raw.get("label", "")),
        cluster_name=str(raw.get("cluster_name", "")),
        resources=resources,
        is_sweep=bool(raw.get("is_sweep", False)),
        sweep_trials=sweep_trials,
        config=dict(raw.get("config", {})),
    )

    backend = get_backend(spec.backend)
    record = backend.submit(client._config.data_root, spec)

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
