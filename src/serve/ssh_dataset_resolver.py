"""Dataset resolution for SSH job submission.

Ensures datasets exist on the remote cluster before launching training.
Auto-pushes from local if the dataset hasn't been pushed yet.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from core.errors import CrucibleRemoteError
from core.job_types import JobSpec
from core.slurm_types import ClusterConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from serve.ssh_connection import SshSession


def resolve_dataset(
    session: SshSession,
    cluster: ClusterConfig,
    data_root: Path,
    spec: JobSpec,
    phase: Callable[[str], None] | None = None,
) -> None:
    """Ensure the dataset exists on the remote and set raw_data_path.

    If the dataset hasn't been pushed yet, auto-pushes it from local.
    Mutates spec.method_args in place to add raw_data_path and the
    method-specific data path field (e.g. lora_data_path for lora-train).
    """
    from core.constants import sanitize_remote_name
    from core.training_methods import DATA_PATH_FIELDS
    from serve.remote_dataset_ops import SOURCE_DATA_FILE_NAME
    from serve.remote_job_submitter import _find_remote_data_file

    ds_name = str(spec.method_args.get("dataset_name", ""))
    if not ds_name:
        return

    workspace = cluster.remote_workspace or "~/crucible-jobs"
    resolved_ws = session.resolve_path(workspace)
    safe_ds = sanitize_remote_name(ds_name)
    ds_path = f"{resolved_ws}/datasets/{safe_ds}"

    _, _, rc = session.execute(f"test -d {shlex.quote(ds_path)}", timeout=10)
    if rc != 0:
        # Dataset not on remote — push it
        if phase:
            phase(f"Pushing dataset '{ds_name}'...")
        from serve.remote_dataset_ops import push_dataset
        push_dataset(session, cluster, ds_name, data_root)

    data_file = _find_remote_data_file(session, ds_path)
    spec.method_args["raw_data_path"] = data_file
    spec.method_args["dataset_path"] = data_file

    # Data-path methods (SFT, LoRA, DPO, etc.) need the method-specific
    # data path field set.  Prefer the original source file (prompt/response
    # format) over crucible ingest records.
    source_file = f"{ds_path}/{SOURCE_DATA_FILE_NAME}"
    _, _, src_rc = session.execute(f"test -f {shlex.quote(source_file)}", timeout=10)
    best_data_file = source_file if src_rc == 0 else data_file

    data_field = DATA_PATH_FIELDS.get(spec.job_type)
    if data_field:
        spec.method_args[data_field] = best_data_file

    # RLHF uses a nested reward_config.preference_data_path instead of
    # a top-level data path field.
    if spec.job_type == "rlhf-train":
        rc = spec.method_args.get("reward_config")
        if isinstance(rc, dict):
            rc["preference_data_path"] = best_data_file

    # Resolve dual datasets for steering compute (positive/negative).
    for ds_key, path_key in (
        ("positive_dataset", "positive_raw_data_path"),
        ("negative_dataset", "negative_raw_data_path"),
    ):
        ds = str(spec.method_args.get(ds_key, ""))
        if ds:
            safe_steer = sanitize_remote_name(ds)
            steer_path = f"{resolved_ws}/datasets/{safe_steer}"
            _, _, steer_rc = session.execute(f"test -d {shlex.quote(steer_path)}", timeout=10)
            if steer_rc != 0:
                if phase:
                    phase(f"Pushing dataset '{ds}'...")
                from serve.remote_dataset_ops import push_dataset
                push_dataset(session, cluster, ds, data_root)
            spec.method_args[path_key] = _find_remote_data_file(
                session, steer_path,
            )
