"""Push a local model to a remote cluster via SSH/SFTP.

Mirrors the dataset push pattern: tar the model directory, upload,
extract on the remote, and return the remote path.
"""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

from core.errors import CrucibleRemoteError
from core.slurm_types import ClusterConfig
from serve.ssh_connection import SshSession


def push_model_to_cluster(
    session: SshSession,
    cluster: ClusterConfig,
    model_name: str,
    model_path: Path,
) -> str:
    """Push a local model directory to the remote cluster.

    Args:
        session: Active SSH session.
        cluster: Target cluster configuration.
        model_name: Registry name of the model.
        model_path: Local path to the model files.

    Returns:
        The remote path where the model was placed.
    """
    if not model_path.exists():
        raise CrucibleRemoteError(f"Local model path not found: {model_path}")

    remote_dir = f"{cluster.remote_workspace}/models/{model_name}"
    session.mkdir_p(remote_dir)

    if model_path.is_file():
        remote_file = f"{remote_dir}/{model_path.name}"
        print(f"Uploading {model_path.name}...", flush=True)
        session.upload(model_path, remote_file)
    else:
        _upload_directory(session, model_path, remote_dir)

    print(f"Model '{model_name}' pushed to {cluster.name}:{remote_dir}", flush=True)
    return remote_dir


def _upload_directory(
    session: SshSession,
    local_dir: Path,
    remote_dir: str,
) -> None:
    """Tar a local directory, upload, and extract on remote."""
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "model.tar.gz"
        print(f"Compressing {local_dir.name}/...", flush=True)
        with tarfile.open(tar_path, "w:gz") as tar:
            for item in local_dir.iterdir():
                tar.add(str(item), arcname=item.name)

        print("Uploading...", flush=True)
        remote_tar = f"{remote_dir}/model.tar.gz"
        session.upload(tar_path, remote_tar)

    print("Extracting on remote...", flush=True)
    _, stderr, code = session.execute(
        f"tar -xzf {remote_tar} -C {remote_dir} && rm -f {remote_tar}",
    )
    if code != 0:
        raise CrucibleRemoteError(
            f"Remote tar extraction failed (exit {code}): {stderr}"
        )
