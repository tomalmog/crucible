"""Download HuggingFace models to remote clusters via SSH.

Connects to a registered cluster, ensures ``huggingface_hub`` is
available in the forge conda env, runs ``snapshot_download`` on
the remote, and registers the model in the local registry.
"""

from __future__ import annotations

import re
from pathlib import Path

from core.errors import ForgeRemoteError
from serve.remote_env_setup import ENV_NAME, conda_cmd, ensure_remote_env
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
        data_root: Root .forge directory.
        repo_id: HuggingFace repo ID (e.g. 'meta-llama/Llama-2-7b').
        cluster_name: Name of the registered cluster.
        model_name: Name for model registry (defaults to repo_id).
        revision: Optional model revision/branch.
        register: Whether to register in the local model registry.

    Returns:
        The remote path where the model was downloaded.

    Raises:
        ForgeRemoteError: If connection, install, or download fails.
    """
    cluster = load_cluster(data_root, cluster_name)

    with SshSession(cluster) as session:
        _progress("Connecting to cluster...")

        ensure_remote_env(session)
        _ensure_hf_hub_installed(session)

        remote_path = _build_remote_model_path(
            cluster.remote_workspace, repo_id,
        )
        session.mkdir_p(remote_path)

        _progress(f"Downloading {repo_id}...")
        _run_snapshot_download(session, repo_id, remote_path, revision)
        _progress(f"Download complete: {remote_path}")

    if register:
        from store.model_registry import ModelRegistry

        effective_name = model_name or repo_id
        registry = ModelRegistry(data_root)
        version = registry.register_remote_model(
            model_name=effective_name,
            remote_host=cluster.host,
            remote_path=remote_path,
        )
        _progress(f"Registered as {effective_name}")
        print(f"version_id={version.version_id}", flush=True)

    return remote_path


def _progress(msg: str) -> None:
    """Print a progress line for UI consumption."""
    print(f"DOWNLOAD_REMOTE: {msg}", flush=True)


def _ensure_hf_hub_installed(session: SshSession) -> None:
    """Install ``huggingface_hub`` in the forge env if missing."""
    check = 'import huggingface_hub; print("hf_ok")'
    stdout, _, code = session.execute(
        conda_cmd(f'conda run -n {ENV_NAME} python -c "{check}"'),
        timeout=30,
    )
    if code == 0 and "hf_ok" in stdout:
        return

    _progress("Installing huggingface_hub...")
    _, stderr, code = session.execute(
        conda_cmd(f"conda run -n {ENV_NAME} pip install 'huggingface_hub'"),
        timeout=300,
    )
    if code != 0:
        raise ForgeRemoteError(
            f"Failed to install huggingface_hub: {stderr.strip()}"
        )


def _build_remote_model_path(remote_workspace: str, repo_id: str) -> str:
    """Build the target directory for the model on the remote."""
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", repo_id)
    return f"{remote_workspace}/models/{safe_name}"


def _run_snapshot_download(
    session: SshSession,
    repo_id: str,
    target_dir: str,
    revision: str | None,
) -> None:
    """Execute ``huggingface_hub.snapshot_download`` on the remote."""
    rev_arg = f', revision="{revision}"' if revision else ""
    script = (
        "from huggingface_hub import snapshot_download; "
        f'p = snapshot_download("{repo_id}", '
        f'local_dir="{target_dir}"{rev_arg}); '
        f'print("downloaded_to=" + str(p))'
    )
    stdout, stderr, code = session.execute(
        conda_cmd(f"conda run -n {ENV_NAME} python -c '{script}'"),
        timeout=1800,
    )
    if code != 0:
        raise ForgeRemoteError(
            f"Remote download of {repo_id} failed: {stderr.strip()}"
        )
    if "downloaded_to=" not in stdout:
        raise ForgeRemoteError(
            f"Download succeeded but path not reported. "
            f"stdout: {stdout[:300]}"
        )
