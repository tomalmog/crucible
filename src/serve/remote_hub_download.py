"""Download HuggingFace models and datasets to remote clusters via SSH.

Connects to a registered cluster, ensures ``huggingface_hub`` is
available in the crucible conda env, runs ``snapshot_download`` on
the remote, and optionally registers the model in the local registry.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from core.constants import sanitize_remote_name
from core.errors import CrucibleRemoteError
from core.slurm_types import ClusterConfig
from serve.managed_conda_env import ENV_NAME, managed_conda_command
from serve.remote_env_setup import ensure_remote_env
from serve.remote_runtime import (
    resolve_cluster_runtime,
    runtime_python_command,
    uses_managed_conda_env,
)
from serve.ssh_connection import SshSession
from store.cluster_registry import load_cluster


def download_model_to_cluster(
    data_root: Path,
    repo_id: str,
    cluster_name: str,
    model_name: str = "",
    revision: str | None = None,
    *,
    register: bool = False,
) -> str:
    """Download a HuggingFace model to a remote cluster via SSH.

    Args:
        data_root: Root .crucible directory.
        repo_id: HuggingFace repo ID (e.g. 'meta-llama/Llama-2-7b').
        cluster_name: Name of the registered cluster.
        model_name: Name for model registry (defaults to repo_id).
        revision: Optional model revision/branch.
        register: Whether to register in the local model registry.

    Returns:
        The remote path where the model was downloaded.

    Raises:
        CrucibleRemoteError: If connection, install, or download fails.
    """
    cluster = load_cluster(data_root, cluster_name)

    with SshSession(cluster) as session:
        _progress("Connecting to cluster...")
        cluster = resolve_cluster_runtime(cluster, session)

        _prepare_hub_runtime(session, cluster)
        python_command = None
        if not uses_managed_conda_env(cluster):
            python_command = runtime_python_command(cluster)

        # Resolve ~ so snapshot_download gets an absolute path.
        remote_path = _build_remote_model_path(cluster.remote_workspace, repo_id)
        session.mkdir_p(remote_path)

        _progress(f"Downloading {repo_id}...")
        _run_snapshot_download(session, repo_id, remote_path, revision, python_command)
        _progress(f"Download complete: {remote_path}")

    if register:
        from store.model_registry import ModelRegistry

        effective_name = model_name or repo_id
        registry = ModelRegistry(data_root)
        entry = registry.register_remote_model(
            model_name=effective_name,
            remote_host=cluster.host,
            remote_path=remote_path,
        )
        _progress(f"Registered as {effective_name}")
        print(f"model_name={entry.model_name}", flush=True)

    return remote_path


def download_dataset_to_cluster(
    data_root: Path,
    repo_id: str,
    cluster_name: str,
    revision: str | None = None,
) -> str:
    """Download a HuggingFace dataset to a remote cluster via SSH.

    Args:
        data_root: Root .crucible directory.
        repo_id: HuggingFace dataset repo ID.
        cluster_name: Name of the registered cluster.
        revision: Optional dataset revision/branch.

    Returns:
        The remote path where the dataset was downloaded.

    Raises:
        CrucibleRemoteError: If connection, install, or download fails.
    """
    cluster = load_cluster(data_root, cluster_name)

    with SshSession(cluster) as session:
        _progress("Connecting to cluster...")
        cluster = resolve_cluster_runtime(cluster, session)

        _prepare_hub_runtime(session, cluster)
        python_command = None
        if not uses_managed_conda_env(cluster):
            python_command = runtime_python_command(cluster)

        # Resolve ~ so snapshot_download gets an absolute path.
        remote_path = _build_remote_dataset_path(cluster.remote_workspace, repo_id)
        session.mkdir_p(remote_path)

        _progress(f"Downloading {repo_id}...")
        _run_snapshot_download(
            session, repo_id, remote_path, revision, python_command,
            repo_type="dataset",
        )
        # Write metadata.json so list_remote_datasets discovers it
        safe_name = sanitize_remote_name(repo_id)
        _write_remote_dataset_metadata(session, remote_path, safe_name)
        _progress(f"Download complete: {remote_path}")

    return remote_path


def _write_remote_dataset_metadata(
    session: SshSession, remote_path: str, dataset_name: str,
) -> None:
    """Write a metadata.json sidecar so list_remote_datasets discovers it."""
    import json
    from datetime import datetime, timezone

    size_bytes = 0
    stdout, _, code = session.execute(
        f"du -sb {shlex.quote(remote_path)} 2>/dev/null | cut -f1",
    )
    if code == 0 and stdout.strip().isdigit():
        size_bytes = int(stdout.strip())

    metadata = json.dumps({
        "name": dataset_name,
        "size_bytes": size_bytes,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    })
    meta_path = f"{remote_path}/metadata.json"
    session.upload_text(metadata, meta_path)


def _progress(msg: str) -> None:
    """Print a progress line for UI consumption."""
    print(f"DOWNLOAD_REMOTE: {msg}", flush=True)


def _ensure_hf_hub_installed(session: SshSession) -> None:
    """Install ``huggingface_hub`` in the crucible env if missing."""
    check = 'import huggingface_hub; print("hf_ok")'
    stdout, _, code = session.execute(
        managed_conda_command(session, f'conda run -n {ENV_NAME} python -c "{check}"'),
        timeout=30,
    )
    if code == 0 and "hf_ok" in stdout:
        return

    _progress("Installing huggingface_hub...")
    _, stderr, code = session.execute(
        managed_conda_command(
            session,
            f"conda run -n {ENV_NAME} pip install 'huggingface_hub'",
        ),
        timeout=300,
    )
    if code != 0:
        raise CrucibleRemoteError(
            f"Failed to install huggingface_hub: {stderr.strip()}"
        )


def _prepare_hub_runtime(session: SshSession, cluster: ClusterConfig) -> None:
    """Ensure the configured remote runtime can import ``huggingface_hub``."""
    if uses_managed_conda_env(cluster):
        ensure_remote_env(session)
        _ensure_hf_hub_installed(session)
        return
    _ensure_hf_hub_for_python(session, cluster)


def _ensure_hf_hub_for_python(session: SshSession, cluster: ClusterConfig) -> None:
    """Install ``huggingface_hub`` for an explicit cluster Python path."""
    python_command = runtime_python_command(cluster)
    check = 'import huggingface_hub; print("hf_ok")'
    stdout, _, code = session.execute(
        f"{python_command} -c {shlex.quote(check)}",
        timeout=30,
    )
    if code == 0 and "hf_ok" in stdout:
        return

    _progress("Installing huggingface_hub...")
    _, stderr, code = session.execute(
        f"{python_command} -m pip install 'huggingface_hub'",
        timeout=300,
    )
    if code != 0:
        raise CrucibleRemoteError(
            f"Failed to install huggingface_hub: {stderr.strip()}"
        )


def _build_remote_model_path(remote_workspace: str, repo_id: str) -> str:
    """Build the target directory for the model on the remote."""
    return f"{remote_workspace}/models/{sanitize_remote_name(repo_id)}"


def _build_remote_dataset_path(remote_workspace: str, repo_id: str) -> str:
    """Build the target directory for the dataset on the remote."""
    return f"{remote_workspace}/datasets/{sanitize_remote_name(repo_id)}"


def _run_snapshot_download(
    session: SshSession,
    repo_id: str,
    target_dir: str,
    revision: str | None,
    python_command: str | None = None,
    repo_type: str = "model",
) -> None:
    """Execute ``huggingface_hub.snapshot_download`` on the remote.

    Uploads a temporary Python script rather than embedding parameters
    in a ``python -c`` shell command to avoid injection via repo_id,
    target_dir, or the HF token.
    """
    import json as _json
    import os

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""

    # Build the download script as a proper Python file so that no
    # user-controlled value is ever interpolated into a shell command.
    kwargs: dict[str, str | None] = {
        "repo_id": repo_id,
        "local_dir": target_dir,
    }
    if revision:
        kwargs["revision"] = revision
    if repo_type != "model":
        kwargs["repo_type"] = repo_type
    if hf_token:
        kwargs["token"] = hf_token

    script_body = (
        "import json, os, sys\n"
        "from huggingface_hub import snapshot_download\n"
        f"kwargs = json.loads({_json.dumps(_json.dumps(kwargs))})\n"
        "p = snapshot_download(**kwargs)\n"
        'print("downloaded_to=" + str(p))\n'
    )

    remote_script = f"{target_dir}/_crucible_download.py"
    session.upload_text(script_body, remote_script)

    try:
        stdout, stderr, code = session.execute(
            _snapshot_command(session, remote_script, python_command),
            timeout=1800,
        )
    finally:
        # Clean up the temp script regardless of success/failure
        session.execute(f"rm -f {shlex.quote(remote_script)}", timeout=10)

    if code != 0:
        raise CrucibleRemoteError(
            f"Remote download of {repo_id} failed: {stderr.strip()}"
        )
    if "downloaded_to=" not in stdout:
        raise CrucibleRemoteError(
            f"Download succeeded but path not reported. "
            f"stdout: {stdout[:300]}"
        )


def _snapshot_command(
    session: SshSession,
    remote_script: str,
    python_command: str | None,
) -> str:
    """Build the remote command that executes the uploaded download script."""
    if python_command:
        return f"{python_command} {shlex.quote(remote_script)}"
    return managed_conda_command(
        session,
        f"conda run -n {ENV_NAME} python {shlex.quote(remote_script)}",
    )
