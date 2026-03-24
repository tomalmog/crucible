"""Cancel and pull operations for remote Slurm jobs.

Handles job cancellation via scancel and model download from
remote clusters into the local model registry.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from core.errors import CrucibleRemoteError
from core.slurm_types import RemoteJobRecord
from serve.remote_result_reader import read_remote_result
from serve.ssh_connection import SshSession
from store.cluster_registry import load_cluster


def cancel_remote_job(
    data_root: Path,
    job_id: str,
) -> RemoteJobRecord:
    """Cancel a running remote job via scancel."""
    from store.remote_job_store import load_remote_job, update_remote_job_state

    record = load_remote_job(data_root, job_id)
    cluster = load_cluster(data_root, record.cluster_name)

    with SshSession(cluster) as session:
        _, stderr, code = session.execute(
            f"scancel {record.slurm_job_id}", timeout=15,
        )
        if code != 0:
            raise CrucibleRemoteError(f"scancel failed: {stderr.strip()}")

    return update_remote_job_state(data_root, job_id, "cancelled")


def pull_remote_model(
    data_root: Path,
    job_id: str,
    model_name: str | None = None,
) -> RemoteJobRecord:
    """Download a trained model from the remote cluster.

    Steps: discover model path -> check remote size -> tar on remote ->
    download -> extract locally -> update model registry.
    """
    import sys
    import tarfile

    from store.model_registry import ModelRegistry
    from store.remote_job_store import load_remote_job, update_remote_job_state

    def progress(msg: str) -> None:
        print(f"CRUCIBLE_PULL_PROGRESS: {msg}", flush=True)
        sys.stdout.flush()

    record = load_remote_job(data_root, job_id)

    # Step 1: Always read result.json fresh to get the model path
    progress("Connecting to cluster...")
    cluster = load_cluster(data_root, record.cluster_name)
    local_dir = data_root / "pulled-models" / job_id
    local_dir.mkdir(parents=True, exist_ok=True)

    with SshSession(cluster) as session:
        remote_model = record.model_path_remote
        if not remote_model:
            remote_model = _read_model_path(session, record)
            record = update_remote_job_state(
                data_root, job_id, record.state,
                model_path_remote=remote_model,
            )
        progress(f"Remote model: {remote_model}")

        model_dir = str(PurePosixPath(remote_model).parent)

        _, _, rc = session.execute(
            f"test -d '{model_dir}'", timeout=10,
        )
        if rc != 0:
            raise CrucibleRemoteError(
                f"Model directory not found on cluster: {model_dir}\n"
                f"(model_path from result.json: {remote_model})"
            )

        # Step 2: Check remote size
        stdout, _, _ = session.execute(
            f"du -sh '{model_dir}' 2>/dev/null || echo 'unknown'",
            timeout=30,
        )
        size_str = stdout.strip().split("\t")[0] if stdout.strip() else "unknown"
        progress(f"Remote model size: {size_str}")

        # Step 3: Tar the model directory on the remote
        progress("Compressing model on cluster...")
        tar_name = "model_download.tar.gz"
        remote_tar = f"{record.remote_output_dir}/{tar_name}"
        stdout, stderr, code = session.execute(
            f"tar czf '{remote_tar}' "
            f"--exclude='checkpoints' "
            f"-C '{model_dir}' .",
            timeout=600,
        )
        if code != 0:
            raise CrucibleRemoteError(
                f"Remote tar failed (dir={model_dir}): {stderr.strip()}"
            )

        stdout, _, _ = session.execute(
            f"du -sh '{remote_tar}' 2>/dev/null || echo 'unknown'",
            timeout=15,
        )
        compressed = stdout.strip().split("\t")[0] if stdout.strip() else "?"
        progress(f"Compressed: {compressed}. Downloading...")

        # Step 4: Download
        local_tar = local_dir / tar_name
        session.download(remote_tar, local_tar)
        progress(f"Downloaded to {local_tar}")

        session.execute(f"rm -f '{remote_tar}'", timeout=15)

    # Step 5: Extract locally
    progress("Extracting model locally...")
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=str(local_dir), filter="data")
    local_tar.unlink()

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

    # Check if model already exists and mark as pulled, otherwise register
    try:
        existing = registry.get_model(effective_name)
        if existing.location_type in ("remote", "both"):
            registry.mark_model_pulled(effective_name, model_local_path)
        else:
            registry.register_model(
                model_name=effective_name,
                model_path=model_local_path,
                run_id=job_id,
            )
    except Exception:
        registry.register_model(
            model_name=effective_name,
            model_path=model_local_path,
            run_id=job_id,
        )

    progress("Complete!")
    return update_remote_job_state(
        data_root, job_id, record.state,
        model_path_local=model_local_path,
    )


def pull_remote_model_direct(
    data_root: Path,
    model_name: str,
    remote_host: str,
    remote_path: str,
) -> str:
    """Download a model from a remote cluster using host + path directly.

    Used for models registered without a job record (e.g. hub downloads).
    Returns the local model path.
    """
    import sys
    import tarfile

    from store.cluster_registry import list_clusters
    from store.model_registry import ModelRegistry

    def progress(msg: str) -> None:
        print(f"CRUCIBLE_PULL_PROGRESS: {msg}", flush=True)
        sys.stdout.flush()

    # Find the cluster config matching the remote host
    clusters = list_clusters(data_root)
    cluster = next((c for c in clusters if c.host == remote_host), None)
    if cluster is None:
        raise CrucibleRemoteError(
            f"No registered cluster with host '{remote_host}'. "
            "Register the cluster first."
        )

    local_dir = data_root / "pulled-models" / model_name
    local_dir.mkdir(parents=True, exist_ok=True)

    progress("Connecting to cluster...")
    with SshSession(cluster) as session:
        remote_parent = str(PurePosixPath(remote_path).parent)
        remote_basename = PurePosixPath(remote_path).name

        # Check if it's a file or directory
        _, _, rc_file = session.execute(
            f"test -f '{remote_path}'", timeout=10,
        )
        is_file = rc_file == 0

        if is_file:
            # Download the model file
            stdout, _, _ = session.execute(
                f"du -sh '{remote_path}' 2>/dev/null || echo 'unknown'",
                timeout=30,
            )
            size_str = stdout.strip().split("\t")[0] if stdout.strip() else "unknown"
            progress(f"Remote model size: {size_str}")
            progress("Downloading model file...")
            local_file = local_dir / remote_basename
            session.download(remote_path, local_file)
            model_local_path = str(local_file)

            # Also download sibling metadata files if present
            _download_metadata_files(session, remote_parent, local_dir)
        else:
            # Directory: tar + download + extract
            _, _, rc_dir = session.execute(
                f"test -d '{remote_path}'", timeout=10,
            )
            if rc_dir != 0:
                raise CrucibleRemoteError(
                    f"Remote path not found: {remote_path}"
                )

            stdout, _, _ = session.execute(
                f"du -sh '{remote_path}' 2>/dev/null || echo 'unknown'",
                timeout=30,
            )
            size_str = stdout.strip().split("\t")[0] if stdout.strip() else "unknown"
            progress(f"Remote model size: {size_str}")

            progress("Compressing model on cluster...")
            tar_name = "model_download.tar.gz"
            remote_tar = f"{remote_parent}/{tar_name}"
            _, stderr, code = session.execute(
                f"tar czf '{remote_tar}' "
                f"--exclude='checkpoints' "
                f"-C '{remote_path}' .",
                timeout=600,
            )
            if code != 0:
                raise CrucibleRemoteError(
                    f"Remote tar failed: {stderr.strip()}"
                )

            stdout, _, _ = session.execute(
                f"du -sh '{remote_tar}' 2>/dev/null || echo 'unknown'",
                timeout=15,
            )
            compressed = stdout.strip().split("\t")[0] if stdout.strip() else "?"
            progress(f"Compressed: {compressed}. Downloading...")

            local_tar = local_dir / tar_name
            session.download(remote_tar, local_tar)
            progress(f"Downloaded to {local_tar}")

            session.execute(f"rm -f '{remote_tar}'", timeout=15)

            progress("Extracting model locally...")
            with tarfile.open(local_tar, "r:gz") as tar:
                tar.extractall(path=str(local_dir), filter="data")
            local_tar.unlink()

            model_local_path = str(local_dir)

    progress(f"Extracted to {model_local_path}")

    # Update model registry
    progress("Updating model registry...")
    registry = ModelRegistry(data_root)
    registry.mark_model_pulled(model_name, model_local_path)

    progress("Complete!")
    return model_local_path


_METADATA_FILES = (
    "training_config.json",
    "tokenizer_vocab.json",
    "vocab.json",
    "tokenizer.json",
    "history.json",
    "lora_adapter.pt",
    "lora_adapter_config.json",
)


def _download_metadata_files(
    session: SshSession,
    remote_dir: str,
    local_dir: Path,
) -> None:
    """Download small metadata files from the model directory."""
    for name in _METADATA_FILES:
        remote_file = f"{remote_dir}/{name}"
        _, _, rc = session.execute(f"test -f '{remote_file}'", timeout=5)
        if rc == 0:
            local_file = local_dir / name
            if not local_file.exists():
                try:
                    session.download(remote_file, local_file)
                except Exception:
                    pass  # Non-critical metadata


def _read_model_path(
    session: SshSession,
    record: RemoteJobRecord,
) -> str:
    """Read result.json from the remote and extract the model path."""
    result = read_remote_result(session, record)
    if not result:
        result_path = f"{record.remote_output_dir}/result.json"
        raise CrucibleRemoteError(
            f"No result.json found at {result_path}. "
            "Training may not have completed."
        )
    model_path = result.get("model_path", "")
    if not model_path:
        error_msg = result.get("error", "unknown error")
        raise CrucibleRemoteError(
            f"Job result has no model_path. "
            f"Training status: {result.get('status', '?')}, "
            f"error: {error_msg}"
        )
    return str(model_path)
