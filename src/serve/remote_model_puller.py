"""Cancel and pull operations for remote Slurm jobs.

Handles job cancellation via scancel and model download from
remote clusters into the local model registry.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.errors import ForgeRemoteError
from core.slurm_types import RemoteJobRecord
from serve.ssh_connection import SshSession
from store.cluster_registry import load_cluster


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
