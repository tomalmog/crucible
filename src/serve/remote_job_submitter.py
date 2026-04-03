"""Orchestrate remote Slurm job submission.

Handles the full lifecycle: bundle agent, upload, generate script,
submit via sbatch, save job record.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from core.constants import sanitize_remote_name
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


def _find_remote_data_file(
    session: SshSession, ds_path: str,
) -> str:
    """Find the ingestible data file in a remote dataset directory.

    Checks for ``records.jsonl`` first (pushed datasets), then
    ``source.jsonl``, then any ``.jsonl`` or ``.parquet`` file
    (hub-downloaded datasets).  For HF datasets with multiple files,
    prefers ``train`` split files.  Returns the absolute remote path.

    Raises ``CrucibleRemoteError`` if no data file is found.
    """
    # Prefer records.jsonl (crucible-ingested datasets)
    for candidate in ("records.jsonl", SOURCE_DATA_FILE_NAME):
        path = f"{ds_path}/{candidate}"
        _, _, rc = session.execute(f"test -f {shlex.quote(path)}", timeout=10)
        if rc == 0:
            return path

    # Hub-downloaded datasets: list .jsonl and .parquet files
    stdout, _, rc = session.execute(
        f"find {shlex.quote(ds_path)} -maxdepth 3 -type f"
        r" \( -name '*.jsonl' -o -name '*.parquet' \)"
        " | sort",
        timeout=15,
    )
    candidates = [f for f in stdout.strip().splitlines() if f.strip()]
    if candidates:
        # Prefer train split files (common HF naming: train-*.parquet)
        train_files = [f for f in candidates if "train" in f.lower()]
        return train_files[0] if train_files else candidates[0]

    raise CrucibleRemoteError(
        f"No data file found in {ds_path}. Expected "
        "records.jsonl, source.jsonl, or .parquet/.jsonl files."
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
            # Verify dataset exists on cluster and resolve data file
            ds_name = str(method_args.get("dataset_name", ""))
            if ds_name:
                safe_ds = sanitize_remote_name(ds_name)
                ds_path = f"{cluster.remote_workspace}/datasets/{safe_ds}"
                _, _, rc = session.execute(f"test -d {shlex.quote(ds_path)}", timeout=10)
                if rc != 0:
                    raise CrucibleRemoteError(
                        f"Dataset '{ds_name}' not found on cluster "
                        f"'{cluster_name}'. Push it first with: crucible remote "
                        f"dataset-push --cluster {cluster_name} "
                        f"--dataset {ds_name}"
                    )
                data_file = _find_remote_data_file(session, ds_path)
                method_args["raw_data_path"] = data_file
                method_args["dataset_path"] = data_file
                # Data-path methods (SFT, LoRA, DPO, etc.) need the
                # original source file (prompt/response format), not
                # the crucible ingest records.
                data_field = DATA_PATH_FIELDS.get(training_method)
                if data_field:
                    source_file = f"{ds_path}/{SOURCE_DATA_FILE_NAME}"
                    _, _, src_rc = session.execute(
                        f"test -f {shlex.quote(source_file)}", timeout=10,
                    )
                    if src_rc == 0:
                        method_args[data_field] = source_file
                    else:
                        method_args[data_field] = data_file
            # Resolve ~ in base_model_path so the agent uses an absolute path
            bmp = str(method_args.get("base_model_path", ""))
            if bmp:
                method_args["base_model_path"] = session.resolve_path(bmp)
            # Ensure tokenizer/config companion files exist next to remote model
            _ensure_remote_model_companions(
                session, cluster, data_root, method_args,
            )
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


def submit_remote_eval_job(
    data_root: Path,
    cluster_name: str,
    method_args: dict[str, object],
    resources: SlurmResourceConfig,
    model_name: str = "",
) -> RemoteJobRecord:
    """Submit an evaluation job to a remote Slurm cluster.

    Args:
        data_root: Root .crucible directory.
        cluster_name: Name of the registered cluster.
        method_args: Eval arguments (model_path, benchmarks, max_samples).
        resources: Resource allocation for the job.

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
        "method": "eval",
        "method_args": method_args,
        "result_output": "result.json",
    }

    ts = now_iso()
    record = RemoteJobRecord(
        job_id=job_id,
        slurm_job_id="",
        cluster_name=cluster_name,
        training_method="eval",
        state="submitting",
        submitted_at=ts,
        updated_at=ts,
        remote_output_dir=workdir,
        submit_phase="Preparing submission...",
        model_name=model_name,
    )
    save_remote_job(data_root, record)

    try:
        _update_phase(data_root, job_id, "Connecting to cluster...")
        with SshSession(cluster) as session:
            cluster = _resolve_cluster_workspace(cluster, session)
            workdir = f"{cluster.remote_workspace}/{job_id}"
            update_remote_job_state(
                data_root, job_id, "submitting",
                remote_output_dir=workdir,
            )
            session.mkdir_p(workdir)
            _update_phase(data_root, job_id, "Provisioning environment...")
            ensure_remote_env(session)
            _update_phase(data_root, job_id, "Uploading eval bundle...")
            _upload_bundle(session, tarball, workdir)
            _upload_config(session, config_payload, workdir)
            script = _generate_script(
                cluster, resources, job_id, "eval",
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


def submit_remote_interp_job(
    data_root: Path,
    cluster_name: str,
    interp_method: str,
    method_args: dict[str, object],
    resources: SlurmResourceConfig,
) -> RemoteJobRecord:
    """Submit an interpretability job to a remote Slurm cluster.

    Args:
        data_root: Root .crucible directory.
        cluster_name: Name of the registered cluster.
        interp_method: One of logit-lens, activation-pca, activation-patch.
        method_args: Analysis arguments (model_path, input_text, etc.).
        resources: Resource allocation for the job.

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
        "method": interp_method,
        "method_args": method_args,
        "result_output": "result.json",
    }

    model_name = str(method_args.get("model_path", ""))
    ts = now_iso()
    record = RemoteJobRecord(
        job_id=job_id,
        slurm_job_id="",
        cluster_name=cluster_name,
        training_method=interp_method,
        state="submitting",
        submitted_at=ts,
        updated_at=ts,
        remote_output_dir=workdir,
        submit_phase="Preparing submission...",
        model_name=model_name,
    )
    save_remote_job(data_root, record)

    try:
        _update_phase(data_root, job_id, "Connecting to cluster...")
        with SshSession(cluster) as session:
            cluster = _resolve_cluster_workspace(cluster, session)
            workdir = f"{cluster.remote_workspace}/{job_id}"
            update_remote_job_state(
                data_root, job_id, "submitting",
                remote_output_dir=workdir,
            )
            session.mkdir_p(workdir)
            _update_phase(data_root, job_id, "Provisioning environment...")
            ensure_remote_env(session)
            _update_phase(data_root, job_id, "Uploading interp bundle...")
            _upload_bundle(session, tarball, workdir)
            # Sync tokenizer/config files next to the remote model
            # (only for Crucible .pt models; HF models download from Hub)
            model_path = str(method_args.get("model_path", ""))
            if model_path:
                from serve.hf_model_loader import is_huggingface_model_id

                if not is_huggingface_model_id(model_path):
                    _ensure_interp_model_companions(
                        session, cluster, data_root, model_path,
                    )
            # Resolve dataset to a data file path (same as training jobs)
            ds_name = str(method_args.get("dataset_name", ""))
            if ds_name:
                safe_ds = sanitize_remote_name(ds_name)
                ds_path = f"{cluster.remote_workspace}/datasets/{safe_ds}"
                _, _, rc = session.execute(f"test -d {shlex.quote(ds_path)}", timeout=10)
                if rc != 0:
                    raise CrucibleRemoteError(
                        f"Dataset '{ds_name}' not found on cluster "
                        f"'{cluster_name}'. Push it first with: crucible remote "
                        f"dataset-push --cluster {cluster_name} "
                        f"--dataset {ds_name}"
                    )
                data_file = _find_remote_data_file(session, ds_path)
                method_args["raw_data_path"] = data_file
            # Resolve dual datasets for steering compute
            for ds_key, path_key in (
                ("positive_dataset", "positive_raw_data_path"),
                ("negative_dataset", "negative_raw_data_path"),
            ):
                ds = str(method_args.get(ds_key, ""))
                if ds:
                    safe = sanitize_remote_name(ds)
                    dp = f"{cluster.remote_workspace}/datasets/{safe}"
                    _, _, drc = session.execute(f"test -d {shlex.quote(dp)}", timeout=10)
                    if drc != 0:
                        raise CrucibleRemoteError(
                            f"Dataset '{ds}' not found on cluster "
                            f"'{cluster_name}'."
                        )
                    method_args[path_key] = _find_remote_data_file(session, dp)
            _upload_config(session, config_payload, workdir)
            script = _generate_script(
                cluster, resources, job_id, interp_method,
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
                safe_ds = sanitize_remote_name(ds_name)
                ds_path = f"{cluster.remote_workspace}/datasets/{safe_ds}"
                _, _, rc = session.execute(
                    f"test -d {shlex.quote(ds_path)}", timeout=10,
                )
                if rc != 0:
                    raise CrucibleRemoteError(
                        f"Dataset '{ds_name}' not found on cluster "
                        f"'{cluster_name}'. Push it first."
                    )
                data_file = _find_remote_data_file(session, ds_path)
                source_file = f"{ds_path}/{SOURCE_DATA_FILE_NAME}"
                data_field = DATA_PATH_FIELDS.get(training_method)
                # Check if source file exists for data-path methods
                use_source = False
                if data_field:
                    _, _, src_rc = session.execute(
                        f"test -f {shlex.quote(source_file)}", timeout=10,
                    )
                    use_source = src_rc == 0
                for tc in trial_configs:
                    tc["raw_data_path"] = data_file
                    tc["dataset_path"] = data_file
                    if data_field:
                        tc[data_field] = source_file if use_source else data_file

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


# Companion files to sync alongside remote model weights.
_MODEL_COMPANIONS = (
    "tokenizer_vocab.json",
    "training_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
)


def _ensure_remote_model_companions(
    session: SshSession,
    cluster: ClusterConfig,
    data_root: Path,
    method_args: dict[str, object],
) -> None:
    """Upload missing tokenizer/config files next to a remote base model.

    Checks if the remote model directory has companion files (tokenizer etc.).
    If missing, looks up the model in the local registry and uploads from the
    local model directory. This ensures remote LoRA/SFT training can find
    the tokenizer without requiring a manual re-push.
    """
    base_model = str(method_args.get("base_model_path", ""))
    if not base_model:
        return

    # Resolve ~ so the path matches the already-resolved cluster workspace
    base_model_resolved = session.resolve_path(base_model)
    models_prefix = f"{cluster.remote_workspace}/models/"
    if not base_model_resolved.startswith(models_prefix):
        print(
            f"Companion sync: skipping — path '{base_model_resolved}' "
            f"not under '{models_prefix}'",
            flush=True,
        )
        return

    # If base_model_path points to a file, use the parent directory
    remote_model_dir = base_model_resolved
    _, _, is_file_rc = session.execute(
        f"test -f {shlex.quote(base_model_resolved)}", timeout=10,
    )
    if is_file_rc == 0:
        # It's a file — use parent directory
        remote_model_dir = str(Path(base_model_resolved).parent)

    try:
        _sync_model_companions(
            session, remote_model_dir, models_prefix, data_root,
        )
    except Exception as exc:
        # Don't let companion sync failure block submission
        print(f"Warning: could not sync model companions: {exc}", flush=True)


def _sync_model_companions(
    session: SshSession,
    remote_model_dir: str,
    models_prefix: str,
    data_root: Path,
) -> None:
    """Upload missing companion files from local registry to remote model."""
    # Check if the Crucible tokenizer file already exists on the remote
    _, _, rc = session.execute(
        f"test -f {shlex.quote(remote_model_dir + '/tokenizer_vocab.json')}", timeout=10,
    )
    if rc == 0:
        return  # tokenizer already present

    # Try to find the local model via registry
    model_name = remote_model_dir[len(models_prefix):].rstrip("/")
    print(f"Companion sync: model_name='{model_name}', remote_dir='{remote_model_dir}'", flush=True)
    from store.model_registry import ModelRegistry
    registry = ModelRegistry(data_root)
    entry = registry.get_model(model_name)

    if not entry.model_path:
        print("Companion sync: no model_path in registry entry", flush=True)
        return

    local_dir = Path(entry.model_path)
    if local_dir.is_file():
        local_dir = local_dir.parent
    print(f"Companion sync: local_dir='{local_dir}'", flush=True)

    uploaded = 0
    for name in _MODEL_COMPANIONS:
        local_file = local_dir / name
        if local_file.is_file():
            remote_file = f"{remote_model_dir}/{name}"
            _, _, exists_rc = session.execute(
                f"test -f {shlex.quote(remote_file)}", timeout=10,
            )
            if exists_rc != 0:
                print(f"Uploading {name} to remote model...", flush=True)
                session.upload(local_file, remote_file)
                uploaded += 1
    if uploaded == 0:
        print(f"Companion sync: no files to upload from {local_dir}", flush=True)


def _ensure_interp_model_companions(
    session: SshSession,
    cluster: ClusterConfig,
    data_root: Path,
    model_path: str,
) -> None:
    """Upload missing tokenizer/config files next to a remote model for interp.

    Similar to ``_ensure_remote_model_companions`` but works with the
    ``model_path`` field used by interp jobs (not ``base_model_path``).
    Looks up the model in the local registry to find local companion files.
    """
    remote_path = session.resolve_path(model_path)

    # Determine the remote directory containing the model
    _, _, is_file_rc = session.execute(
        f"test -f {shlex.quote(remote_path)}", timeout=10,
    )
    if is_file_rc == 0:
        remote_model_dir = str(Path(remote_path).parent)
    else:
        remote_model_dir = remote_path

    # Check if tokenizer already exists on remote
    _, _, rc = session.execute(
        f"test -f {shlex.quote(remote_model_dir + '/tokenizer_vocab.json')}", timeout=10,
    )
    if rc == 0:
        return  # already present

    # Find local model directory via registry
    models_prefix = f"{cluster.remote_workspace}/models/"
    if remote_model_dir.startswith(models_prefix):
        model_name = remote_model_dir[len(models_prefix):].rstrip("/")
    else:
        # Try the outputs dir pattern: .../outputs/train/ → use jobId
        model_name = ""

    local_dir: Path | None = None
    if model_name:
        try:
            from store.model_registry import ModelRegistry
            registry = ModelRegistry(data_root)
            entry = registry.get_model(model_name)
            if entry.model_path:
                local_dir = Path(entry.model_path)
                if local_dir.is_file():
                    local_dir = local_dir.parent
        except Exception:
            pass

    if not local_dir:
        # Fallback: check if any local model version matches the remote path
        try:
            from store.model_registry import ModelRegistry
            registry = ModelRegistry(data_root)
            for group in registry.list_models():
                for version in registry.list_versions(group.name):
                    if version.remote_path and remote_path.startswith(
                        session.resolve_path(version.remote_path),
                    ):
                        lp = Path(version.model_path)
                        local_dir = lp.parent if lp.is_file() else lp
                        break
                if local_dir:
                    break
        except Exception:
            pass

    if not local_dir:
        print(
            f"Interp companion sync: no local model found for {model_path}",
            flush=True,
        )
        return

    uploaded = 0
    for name in _MODEL_COMPANIONS:
        local_file = local_dir / name
        if local_file.is_file():
            remote_file = f"{remote_model_dir}/{name}"
            _, _, exists_rc = session.execute(
                f"test -f {shlex.quote(remote_file)}", timeout=10,
            )
            if exists_rc != 0:
                print(f"Uploading {name} to remote model...", flush=True)
                session.upload(local_file, remote_file)
                uploaded += 1
    if uploaded > 0:
        print(
            f"Interp companion sync: uploaded {uploaded} files to {remote_model_dir}",
            flush=True,
        )
