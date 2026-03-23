"""Push a local model to a remote cluster via SSH/SFTP.

Mirrors the dataset push pattern: tar the model directory, upload,
extract on the remote, and return the remote path.
"""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

from core.constants import sanitize_remote_name
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

    remote_dir = f"{cluster.remote_workspace}/models/{sanitize_remote_name(model_name)}"
    session.mkdir_p(remote_dir)

    if model_path.is_file():
        remote_file = f"{remote_dir}/{model_path.name}"
        print(f"Uploading {model_path.name}...", flush=True)
        session.upload(model_path, remote_file)
        # Also upload companion files (tokenizer, config) from the same directory
        _upload_companion_files(session, model_path.parent, remote_dir)
    else:
        _upload_directory(session, model_path, remote_dir)

    print(f"Model '{model_name}' pushed to {cluster.name}:{remote_dir}", flush=True)
    return remote_dir


# Files that should accompany model weights when pushing a single .pt file.
_COMPANION_FILES = (
    "tokenizer_vocab.json",
    "training_config.json",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
)


def _upload_companion_files(
    session: SshSession,
    local_dir: Path,
    remote_dir: str,
) -> None:
    """Upload tokenizer/config files that sit alongside model weights."""
    for name in _COMPANION_FILES:
        path = local_dir / name
        if path.is_file():
            print(f"Uploading {name}...", flush=True)
            session.upload(path, f"{remote_dir}/{name}")


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
