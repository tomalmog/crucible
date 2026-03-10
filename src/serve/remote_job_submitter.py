"""Orchestrate remote Slurm job submission.

Handles the full lifecycle: bundle agent, upload, generate script,
submit via sbatch, save job record.
"""

from __future__ import annotations

from pathlib import Path

from core.errors import CrucibleRemoteError
from core.slurm_types import (
    ClusterConfig,
    RemoteJobRecord,
    SlurmResourceConfig,
)
from core.training_methods import DATA_PATH_FIELDS
from serve.agent_bundler import build_agent_tarball
from serve.remote_dataset_ops import SOURCE_DATA_FILE_NAME
from serve.remote_data_upload import (
    _generate_script,
    _submit_sbatch,
    _upload_bundle,
    _upload_config,
    _upload_script,
)
from serve.remote_env_setup import ensure_remote_env
from serve.slurm_script_gen import generate_sweep_script
from serve.ssh_connection import SshSession
from store.cluster_registry import load_cluster
from store.remote_job_store import (
    generate_job_id,
    now_iso,
    save_remote_job,
    update_remote_job_state,
)


def _resolve_cluster_workspace(
    cluster: ClusterConfig, session: SshSession,
) -> ClusterConfig:
    """Return a copy of *cluster* with remote_workspace as an absolute path."""
    from dataclasses import replace as dc_replace
    resolved = session.resolve_path(cluster.remote_workspace)
    if resolved != cluster.remote_workspace:
        return dc_replace(cluster, remote_workspace=resolved)
    return cluster


def _update_phase(
    data_root: Path,
    job_id: str,
    phase: str,
) -> None:
    """Update the submit_phase field on a job record and print for CLI."""
    print(f"CRUCIBLE_SUBMIT_PHASE: {phase}", flush=True)
    update_remote_job_state(
        data_root, job_id, "submitting", submit_phase=phase,
    )


def submit_remote_job(
    data_root: Path,
    cluster_name: str,
    training_method: str,
    method_args: dict[str, object],
    resources: SlurmResourceConfig,
    pull_model: bool = False,
    model_name: str = "",
) -> RemoteJobRecord:
    """Submit a training job to a remote Slurm cluster.

    Args:
        data_root: Root .crucible directory.
        cluster_name: Name of the registered cluster.
        training_method: Training method to dispatch.
        method_args: Arguments passed to the training method.
        resources: Resource allocation for the job.
        pull_model: Whether to auto-pull model after completion.
        model_name: Name to register model under in registry.

    Returns:
        The persisted RemoteJobRecord.
    """
    cluster = load_cluster(data_root, cluster_name)
    job_id = generate_job_id()
    workdir = f"{cluster.remote_workspace}/{job_id}"

    tarball = build_agent_tarball(
        cache_dir=data_root / "cache" / "agent-bundles",
    )

    config_payload = {
        "method": training_method,
        "method_args": method_args,
        "result_output": "result.json",
    }

    # Write early JSON so the UI picks it up immediately
    ts = now_iso()
    record = RemoteJobRecord(
        job_id=job_id,
        slurm_job_id="",
        cluster_name=cluster_name,
        training_method=training_method,
        state="submitting",
        submitted_at=ts,
        updated_at=ts,
        remote_output_dir=workdir,
        model_name=model_name,
        submit_phase="Preparing submission...",
    )
    save_remote_job(data_root, record)

    try:
        _update_phase(data_root, job_id, "Connecting to cluster...")
        with SshSession(cluster) as session:
            # Resolve ~ to absolute path so Slurm/SFTP work correctly
            cluster = _resolve_cluster_workspace(cluster, session)
            workdir = f"{cluster.remote_workspace}/{job_id}"
            update_remote_job_state(
                data_root, job_id, "submitting",
                remote_output_dir=workdir,
            )
            session.mkdir_p(workdir)
            _update_phase(data_root, job_id, "Provisioning environment...")
            ensure_remote_env(session)
            _update_phase(data_root, job_id, "Uploading training bundle...")
            _upload_bundle(session, tarball, workdir)
            # Verify dataset exists on cluster if specified
            ds_name = str(method_args.get("dataset_name", ""))
            if ds_name:
                ds_path = f"{cluster.remote_workspace}/datasets/{ds_name}"
                _, _, rc = session.execute(f"test -d {ds_path}", timeout=10)
                if rc != 0:
                    raise CrucibleRemoteError(
                        f"Dataset '{ds_name}' not found on cluster "
                        f"'{cluster_name}'. Push it first with: crucible remote "
                        f"dataset-push --cluster {cluster_name} "
                        f"--dataset {ds_name}"
                    )
                records_file = f"{ds_path}/records.jsonl"
                source_file = f"{ds_path}/{SOURCE_DATA_FILE_NAME}"
                method_args["raw_data_path"] = records_file
                method_args["dataset_path"] = records_file
                # Data-path methods (SFT, LoRA, DPO, etc.) need the
                # original source file (prompt/response format), not
                # the crucible ingest records.
                data_field = DATA_PATH_FIELDS.get(training_method)
                if data_field:
                    _, _, src_rc = session.execute(
                        f"test -f {source_file}", timeout=10,
                    )
                    if src_rc == 0:
                        method_args[data_field] = source_file
                    else:
                        method_args[data_field] = records_file
            _upload_config(session, config_payload, workdir)
            script = _generate_script(
                cluster, resources, job_id, training_method,
            )
            _upload_script(session, script, workdir)
            _update_phase(data_root, job_id, "Submitting to Slurm...")
            slurm_job_id = _submit_sbatch(session, workdir)
    except Exception as exc:
        update_remote_job_state(
            data_root, job_id, "failed",
            submit_phase=f"Failed: {exc}",
        )
        raise

    return update_remote_job_state(
        data_root, job_id, "running",
        slurm_job_id=slurm_job_id,
        remote_log_path=f"{workdir}/slurm-{slurm_job_id}.out",
        submit_phase="",
    )


def submit_remote_sweep(
    data_root: Path,
    cluster_name: str,
    training_method: str,
    trial_configs: list[dict[str, object]],
    resources: SlurmResourceConfig,
) -> RemoteJobRecord:
    """Submit a sweep as a Slurm job array.

    Args:
        data_root: Root .crucible directory.
        cluster_name: Registered cluster name.
        training_method: Training method for each trial.
        trial_configs: List of per-trial method_args dicts.
        resources: Resource allocation per trial.

    Returns:
        The persisted RemoteJobRecord for the sweep.
    """
    cluster = load_cluster(data_root, cluster_name)
    job_id = generate_job_id()
    workdir = f"{cluster.remote_workspace}/{job_id}"
    array_size = len(trial_configs)

    tarball = build_agent_tarball(
        cache_dir=data_root / "cache" / "agent-bundles",
    )

    # Write early JSON so the UI picks it up immediately
    ts = now_iso()
    record = RemoteJobRecord(
        job_id=job_id,
        slurm_job_id="",
        cluster_name=cluster_name,
        training_method=training_method,
        state="submitting",
        submitted_at=ts,
        updated_at=ts,
        remote_output_dir=workdir,
        is_sweep=True,
        sweep_array_size=array_size,
        submit_phase="Preparing submission...",
    )
    save_remote_job(data_root, record)

    try:
        _update_phase(data_root, job_id, "Connecting to cluster...")
        with SshSession(cluster) as session:
            # Resolve ~ to absolute path so Slurm/SFTP work correctly
            cluster = _resolve_cluster_workspace(cluster, session)
            workdir = f"{cluster.remote_workspace}/{job_id}"
            update_remote_job_state(
                data_root, job_id, "submitting",
                remote_output_dir=workdir,
            )
            session.mkdir_p(workdir)
            session.mkdir_p(f"{workdir}/trials")
            _update_phase(data_root, job_id, "Provisioning environment...")
            ensure_remote_env(session)
            _update_phase(data_root, job_id, "Uploading training bundle...")
            _upload_bundle(session, tarball, workdir)
            # Verify dataset exists on cluster if specified
            ds_name = str(
                (trial_configs[0] if trial_configs else {}).get(
                    "dataset_name", "",
                ),
            )
            if ds_name:
                ds_path = f"{cluster.remote_workspace}/datasets/{ds_name}"
                _, _, rc = session.execute(
                    f"test -d {ds_path}", timeout=10,
                )
                if rc != 0:
                    raise CrucibleRemoteError(
                        f"Dataset '{ds_name}' not found on cluster "
                        f"'{cluster_name}'. Push it first."
                    )
                records_file = f"{ds_path}/records.jsonl"
                source_file = f"{ds_path}/{SOURCE_DATA_FILE_NAME}"
                data_field = DATA_PATH_FIELDS.get(training_method)
                # Check if source file exists for data-path methods
                use_source = False
                if data_field:
                    _, _, src_rc = session.execute(
                        f"test -f {source_file}", timeout=10,
                    )
                    use_source = src_rc == 0
                for tc in trial_configs:
                    tc["raw_data_path"] = records_file
                    tc["dataset_path"] = records_file
                    if data_field:
                        tc[data_field] = source_file if use_source else records_file

            for i, trial_args in enumerate(trial_configs):
                config_payload = {
                    "method": training_method,
                    "method_args": trial_args,
                    "result_output": f"result_trial_{i}.json",
                }
                _upload_config(
                    session, config_payload, workdir,
                    filename=f"trials/trial_{i}.json",
                )

            script = generate_sweep_script(
                cluster, resources, job_id, training_method, array_size,
            )
            _upload_script(session, script, workdir)
            _update_phase(data_root, job_id, "Submitting to Slurm...")
            slurm_job_id = _submit_sbatch(session, workdir)
    except Exception as exc:
        update_remote_job_state(
            data_root, job_id, "failed",
            submit_phase=f"Failed: {exc}",
        )
        raise

    return update_remote_job_state(
        data_root, job_id, "running",
        slurm_job_id=slurm_job_id,
        submit_phase="",
    )
