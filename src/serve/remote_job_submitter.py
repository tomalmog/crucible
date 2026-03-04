"""Orchestrate remote Slurm job submission.

Handles the full lifecycle: bundle agent, upload, generate script,
submit via sbatch, save job record.
"""

from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

from core.constants import CATALOG_FILE_NAME, DATASETS_DIR_NAME, VERSIONS_DIR_NAME
from core.errors import ForgeRemoteError
from core.slurm_types import (
    ClusterConfig,
    DataStrategy,
    RemoteJobRecord,
    SlurmResourceConfig,
)
from core.training_methods import DATA_PATH_FIELDS
from serve.agent_bundler import build_agent_tarball
from serve.slurm_script_gen import (
    generate_multi_node_script,
    generate_single_node_script,
    generate_sweep_script,
)
from serve.ssh_connection import SshSession
from store.cluster_registry import load_cluster
from store.remote_job_store import generate_job_id, now_iso, save_remote_job


def submit_remote_job(
    data_root: Path,
    cluster_name: str,
    training_method: str,
    method_args: dict[str, object],
    resources: SlurmResourceConfig,
    data_strategy: DataStrategy = "shared",
    dataset_path: str = "",
    pull_model: bool = False,
    model_name: str = "",
) -> RemoteJobRecord:
    """Submit a training job to a remote Slurm cluster.

    Args:
        data_root: Root .forge directory.
        cluster_name: Name of the registered cluster.
        training_method: Training method to dispatch.
        method_args: Arguments passed to the training method.
        resources: Resource allocation for the job.
        data_strategy: How to provide data to the remote.
        dataset_path: Local dataset path (for scp strategy).
        pull_model: Whether to auto-pull model after completion.

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

    with SshSession(cluster) as session:
        session.mkdir_p(workdir)
        _upload_bundle(session, tarball, workdir)
        _handle_data_strategy(
            session, data_strategy, dataset_path, method_args, workdir,
            training_method, data_root,
        )
        # Upload config AFTER data strategy so rewritten paths are included
        _upload_config(session, config_payload, workdir)
        script = _generate_script(
            cluster, resources, job_id, training_method,
        )
        _upload_script(session, script, workdir)
        slurm_job_id = _submit_sbatch(session, workdir)

    record = RemoteJobRecord(
        job_id=job_id,
        slurm_job_id=slurm_job_id,
        cluster_name=cluster_name,
        training_method=training_method,
        state="running",
        submitted_at=now_iso(),
        updated_at=now_iso(),
        remote_output_dir=workdir,
        remote_log_path=f"{workdir}/slurm-{slurm_job_id}.out",
        model_name=model_name,
    )
    save_remote_job(data_root, record)
    return record


def submit_remote_sweep(
    data_root: Path,
    cluster_name: str,
    training_method: str,
    trial_configs: list[dict[str, object]],
    resources: SlurmResourceConfig,
    data_strategy: DataStrategy = "shared",
    dataset_path: str = "",
) -> RemoteJobRecord:
    """Submit a sweep as a Slurm job array.

    Args:
        data_root: Root .forge directory.
        cluster_name: Registered cluster name.
        training_method: Training method for each trial.
        trial_configs: List of per-trial method_args dicts.
        resources: Resource allocation per trial.
        data_strategy: How to provide data to the remote.
        dataset_path: Local dataset path (for scp strategy).

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

    with SshSession(cluster) as session:
        session.mkdir_p(workdir)
        session.mkdir_p(f"{workdir}/trials")
        _upload_bundle(session, tarball, workdir)

        # Handle data strategy first so paths are rewritten
        _handle_data_strategy(
            session, data_strategy, dataset_path,
            trial_configs[0] if trial_configs else {},
            workdir, training_method, data_root,
        )
        # Apply rewritten data path to all trial configs
        data_field = DATA_PATH_FIELDS.get(training_method)
        if data_field and trial_configs:
            rewritten = trial_configs[0].get(data_field)
            if rewritten:
                for tc in trial_configs[1:]:
                    tc[data_field] = rewritten

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
        slurm_job_id = _submit_sbatch(session, workdir)

    record = RemoteJobRecord(
        job_id=job_id,
        slurm_job_id=slurm_job_id,
        cluster_name=cluster_name,
        training_method=training_method,
        state="running",
        submitted_at=now_iso(),
        updated_at=now_iso(),
        remote_output_dir=workdir,
        is_sweep=True,
        sweep_array_size=array_size,
    )
    save_remote_job(data_root, record)
    return record


def cancel_remote_job(
    data_root: Path,
    job_id: str,
) -> RemoteJobRecord:
    """Cancel a running remote job via scancel.

    Args:
        data_root: Root .forge directory.
        job_id: Local job identifier.

    Returns:
        Updated RemoteJobRecord.
    """
    from store.remote_job_store import load_remote_job, update_remote_job_state

    record = load_remote_job(data_root, job_id)
    cluster = load_cluster(data_root, record.cluster_name)

    with SshSession(cluster) as session:
        _, stderr, code = session.execute(
            f"scancel {record.slurm_job_id}", timeout=15,
        )
        if code != 0:
            raise ForgeRemoteError(f"scancel failed: {stderr.strip()}")

    return update_remote_job_state(data_root, job_id, "cancelled")


def pull_remote_model(
    data_root: Path,
    job_id: str,
    model_name: str | None = None,
) -> RemoteJobRecord:
    """Download a trained model from the remote cluster.

    Steps: discover model path → check remote size → tar on remote →
    download → extract locally → update model registry.

    Prints progress messages to stdout for UI consumption.

    Args:
        data_root: Root .forge directory.
        job_id: Local job identifier.
        model_name: Name to register under. Auto-generated if None.

    Returns:
        Updated RemoteJobRecord with local model path.
    """
    import sys
    import tarfile

    from store.model_registry import ModelRegistry
    from store.remote_job_store import load_remote_job, update_remote_job_state

    def progress(msg: str) -> None:
        print(f"FORGE_PULL_PROGRESS: {msg}", flush=True)
        sys.stdout.flush()

    record = load_remote_job(data_root, job_id)

    # Step 1: Discover model path if not set
    progress("Connecting to cluster...")
    if not record.model_path_remote:
        cluster = load_cluster(data_root, record.cluster_name)
        with SshSession(cluster) as session:
            result_path = f"{record.remote_output_dir}/result.json"
            stdout, _, code = session.execute(
                f"cat {result_path}", timeout=15,
            )
            if code != 0:
                raise ForgeRemoteError("No model found for this job.")
            result = json.loads(stdout)
            model_path = result.get("model_path", "")
            if not model_path:
                raise ForgeRemoteError("Job result has no model_path.")
            record = update_remote_job_state(
                data_root, job_id, record.state,
                model_path_remote=model_path,
            )

    remote_model = record.model_path_remote
    progress(f"Remote model: {remote_model}")

    # Step 2: Check remote size
    cluster = load_cluster(data_root, record.cluster_name)
    local_dir = data_root / "pulled-models" / job_id
    local_dir.mkdir(parents=True, exist_ok=True)

    with SshSession(cluster) as session:
        # Tar the model's parent directory to get all artifacts
        # (model.pt, training_config.json, tokenizer_vocab.json, etc.)
        model_dir = f"$(dirname {remote_model})"
        stdout, _, _ = session.execute(
            f"du -sh {model_dir} 2>/dev/null || echo 'unknown'",
            timeout=30,
        )
        size_str = stdout.strip().split("\t")[0] if stdout.strip() else "unknown"
        progress(f"Remote model size: {size_str}")

        # Step 3: Tar the model directory on the remote
        progress("Compressing model on cluster...")
        tar_name = "model_download.tar.gz"
        remote_tar = f"{record.remote_output_dir}/{tar_name}"
        stdout, stderr, code = session.execute(
            f"tar czf {remote_tar} "
            f"--exclude='checkpoints' "
            f"-C {model_dir} .",
            timeout=600,
        )
        if code != 0:
            raise ForgeRemoteError(f"Remote tar failed: {stderr.strip()}")

        # Get compressed size
        stdout, _, _ = session.execute(
            f"du -sh {remote_tar} 2>/dev/null || echo 'unknown'",
            timeout=15,
        )
        compressed = stdout.strip().split("\t")[0] if stdout.strip() else "?"
        progress(f"Compressed: {compressed}. Downloading...")

        # Step 4: Download
        local_tar = local_dir / tar_name
        session.download(remote_tar, local_tar)
        progress(f"Downloaded to {local_tar}")

        # Clean up remote tar
        session.execute(f"rm -f {remote_tar}", timeout=15)

    # Step 5: Extract locally
    progress("Extracting model locally...")
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=str(local_dir), filter="data")
    local_tar.unlink()

    # Resolve actual model path — tarball extracts into local_dir
    remote_basename = Path(remote_model).name
    candidate = local_dir / remote_basename
    if candidate.is_file():
        model_local_path = str(candidate)
    elif candidate.is_dir():
        model_local_path = str(candidate)
    else:
        model_local_path = str(local_dir)
    progress(f"Extracted to {model_local_path}")

    # Step 6: Update model registry
    progress("Updating model registry...")
    registry = ModelRegistry(data_root)
    effective_name = (
        model_name
        or record.model_name
        or f"remote-{record.training_method}-{job_id[:16]}"
    )

    # Check if a remote version already exists for this job
    existing_version_id = record.local_version_id
    if existing_version_id:
        try:
            registry.mark_model_pulled(existing_version_id, model_local_path)
            version_id = existing_version_id
        except Exception:
            version = registry.register_model(
                model_name=effective_name,
                model_path=model_local_path,
                run_id=job_id,
            )
            version_id = version.version_id
    else:
        # Find remote version by run_id match, or register fresh
        found = _find_version_by_run_id(registry, data_root, job_id)
        if found:
            registry.mark_model_pulled(found, model_local_path)
            version_id = found
        else:
            version = registry.register_model(
                model_name=effective_name,
                model_path=model_local_path,
                run_id=job_id,
            )
            version_id = version.version_id

    progress("Complete!")
    return update_remote_job_state(
        data_root, job_id, record.state,
        model_path_local=model_local_path,
        local_version_id=version_id,
    )


def _find_version_by_run_id(
    registry: object,
    data_root: Path,
    run_id: str,
) -> str | None:
    """Find a model version registered with the given run_id."""
    from store.model_registry_io import load_model_version

    versions_dir = data_root / "models" / "versions"
    if not versions_dir.is_dir():
        return None
    for path in versions_dir.glob("mv-*.json"):
        try:
            version = load_model_version(versions_dir.parent, path.stem)
            if version.run_id == run_id:
                return version.version_id
        except Exception:
            continue
    return None


def _upload_bundle(session: SshSession, tarball: Path, workdir: str) -> None:
    """Upload the agent tarball to the remote workspace."""
    session.upload(tarball, f"{workdir}/forge-agent.tar.gz")


def _upload_config(
    session: SshSession,
    config: dict[str, object],
    workdir: str,
    filename: str = "training_config.json",
) -> None:
    """Write and upload a training config JSON file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False,
    ) as f:
        json.dump(config, f, indent=2)
        tmp_path = Path(f.name)
    try:
        session.upload(tmp_path, f"{workdir}/{filename}")
    finally:
        tmp_path.unlink(missing_ok=True)


def _upload_script(session: SshSession, script: str, workdir: str) -> None:
    """Write and upload an sbatch script."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False,
    ) as f:
        f.write(script)
        tmp_path = Path(f.name)
    try:
        session.upload(tmp_path, f"{workdir}/job.sh")
    finally:
        tmp_path.unlink(missing_ok=True)


def _submit_sbatch(session: SshSession, workdir: str) -> str:
    """Submit the sbatch script and parse the Slurm job ID."""
    stdout, stderr, code = session.execute(
        f"sbatch {workdir}/job.sh", timeout=30,
    )
    if code != 0:
        raise ForgeRemoteError(f"sbatch failed: {stderr.strip()}")
    # sbatch output: "Submitted batch job 12345"
    parts = stdout.strip().split()
    if len(parts) < 4:
        raise ForgeRemoteError(f"Unexpected sbatch output: {stdout.strip()}")
    return parts[-1]


def _generate_script(
    cluster: ClusterConfig,
    resources: SlurmResourceConfig,
    job_id: str,
    training_method: str,
) -> str:
    """Select and generate the appropriate sbatch script."""
    if resources.nodes > 1:
        return generate_multi_node_script(
            cluster, resources, job_id, training_method,
        )
    return generate_single_node_script(
        cluster, resources, job_id, training_method,
    )


_RECORD_BASED_METHODS = frozenset({"train", "distill", "domain-adapt"})


def _upload_dataset_catalog(
    session: SshSession,
    data_root: Path,
    dataset_name: str,
    workdir: str,
) -> None:
    """Upload dataset catalog and latest version to remote.

    Reads the local catalog, builds a minimal single-version catalog,
    tars the latest version directory (manifest + records), and
    extracts into ``{workdir}/.forge/datasets/{name}/``.

    Args:
        session: Active SSH session.
        data_root: Local ``.forge`` root directory.
        dataset_name: Dataset identifier.
        workdir: Remote job working directory.
    """
    from store.catalog_io import read_catalog_file

    ds_dir = data_root / DATASETS_DIR_NAME / dataset_name
    catalog_path = ds_dir / CATALOG_FILE_NAME
    catalog = read_catalog_file(catalog_path)

    latest_id = catalog.get("latest_version") or ""
    if not latest_id:
        raise ForgeRemoteError(
            f"Dataset '{dataset_name}' catalog has no latest version.",
        )

    version_dir = ds_dir / VERSIONS_DIR_NAME / latest_id
    if not version_dir.is_dir():
        raise ForgeRemoteError(
            f"Latest version directory missing: {version_dir}",
        )

    # Find the catalog entry for the latest version
    latest_entry = None
    for entry in catalog.get("versions", []):
        if entry.get("version_id") == latest_id:
            latest_entry = entry
            break
    if not latest_entry:
        raise ForgeRemoteError(
            f"Version '{latest_id}' not found in catalog entries.",
        )

    # Build minimal catalog with only the latest version
    minimal_catalog = {
        "latest_version": latest_id,
        "versions": [latest_entry],
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Write minimal catalog
        catalog_tmp = tmp_path / CATALOG_FILE_NAME
        catalog_tmp.write_text(
            json.dumps(minimal_catalog, indent=2) + "\n",
            encoding="utf-8",
        )

        # Create tarball: catalog.json + versions/{id}/*
        tar_path = tmp_path / "dataset_catalog.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(str(catalog_tmp), arcname=CATALOG_FILE_NAME)
            for child in version_dir.iterdir():
                # Skip Lance dir — remote only needs JSONL
                if child.name == "data.lance":
                    continue
                tar.add(
                    str(child),
                    arcname=f"{VERSIONS_DIR_NAME}/{latest_id}/{child.name}",
                )

        remote_ds = f"{workdir}/.forge/{DATASETS_DIR_NAME}/{dataset_name}"
        session.mkdir_p(remote_ds)
        session.upload(tar_path, f"{remote_ds}/dataset_catalog.tar.gz")
        session.execute(
            f"cd {remote_ds} && tar xzf dataset_catalog.tar.gz"
            " && rm dataset_catalog.tar.gz",
        )


def _handle_data_strategy(
    session: SshSession,
    strategy: DataStrategy,
    dataset_path: str,
    method_args: dict[str, object],
    workdir: str,
    training_method: str = "",
    data_root: Path | None = None,
) -> None:
    """Handle data transfer based on the chosen strategy.

    For record-based methods with a ``dataset_name``, uploads the
    dataset catalog and latest version so the remote agent can call
    ``client.train(dataset_name=...)`` directly.

    For ``scp`` strategy, uploads the dataset and rewrites the
    appropriate data path field in *method_args* so the remote
    entry script uses the uploaded path instead of the local one.
    """
    ds_name = str(method_args.get("dataset_name", ""))
    if ds_name and training_method in _RECORD_BASED_METHODS and data_root:
        _upload_dataset_catalog(session, data_root, ds_name, workdir)
        return

    if strategy == "scp" and dataset_path:
        local_data = Path(dataset_path)
        if local_data.is_file():
            remote_data = f"{workdir}/data/{local_data.name}"
            session.mkdir_p(f"{workdir}/data")
            session.upload(local_data, remote_data)
        elif local_data.is_dir():
            with tempfile.NamedTemporaryFile(
                suffix=".tar.gz", delete=False,
            ) as f:
                tmp_tar = Path(f.name)
            with tarfile.open(tmp_tar, "w:gz") as tar:
                tar.add(str(local_data), arcname=local_data.name)
            session.mkdir_p(f"{workdir}/data")
            session.upload(tmp_tar, f"{workdir}/data/dataset.tar.gz")
            session.execute(
                f"cd {workdir}/data && tar xzf dataset.tar.gz",
            )
            tmp_tar.unlink(missing_ok=True)
            remote_data = f"{workdir}/data/{local_data.name}"
        else:
            return

        # Rewrite data path in method_args to point to uploaded file
        field = DATA_PATH_FIELDS.get(training_method)
        if field:
            method_args[field] = remote_data
        else:
            # train/distill/domain-adapt use raw_data_path for remote
            method_args["raw_data_path"] = remote_data
    # shared and s3 strategies: data path is already in method_args
