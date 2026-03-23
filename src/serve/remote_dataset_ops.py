"""Remote dataset operations over SSH.

Provides push, list, pull, and delete for syncing local
datasets to remote Slurm clusters via SshSession.
"""

from __future__ import annotations

import json
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.constants import (
    DATASETS_DIR_NAME,
    MANIFEST_FILE_NAME,
    RECORDS_FILE_NAME,
    sanitize_remote_name,
)
from core.errors import CrucibleRemoteError
from core.slurm_types import ClusterConfig
from serve.ssh_connection import SshSession
from store.catalog_io import read_manifest_file


@dataclass(frozen=True)
class RemoteDatasetInfo:
    """Metadata for a dataset synced to a remote cluster."""

    name: str
    size_bytes: int
    synced_at: str


def _remote_datasets_dir(cluster: ClusterConfig) -> str:
    """Return the remote datasets base directory."""
    return f"{cluster.remote_workspace}/{DATASETS_DIR_NAME}"


def _build_metadata(
    name: str,
    size_bytes: int,
) -> dict[str, Any]:
    """Build metadata dict for the remote dataset."""
    return {
        "name": name,
        "size_bytes": size_bytes,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }


def _find_source_file(dataset_dir: Path) -> Path | None:
    """Find the original source data file from the dataset's manifest."""
    manifest_path = dataset_dir / MANIFEST_FILE_NAME
    if not manifest_path.exists():
        return None
    manifest = read_manifest_file(manifest_path)
    uri = manifest.source_uri
    if uri and not uri.startswith("s3://"):
        p = Path(uri).expanduser().resolve()
        if p.is_file():
            return p
    return None


def push_dataset(
    session: SshSession,
    cluster: ClusterConfig,
    dataset_name: str,
    data_root: Path,
) -> RemoteDatasetInfo:
    """Push a local dataset to the remote cluster.

    Reads local records directly from datasets/{name}/records.jsonl,
    uploads with a metadata sidecar, and extracts on remote.
    Also uploads the original source data file (for SFT/LoRA/DPO
    training methods that need prompt/response format).
    """
    dataset_dir = data_root / DATASETS_DIR_NAME / dataset_name
    records_path = dataset_dir / RECORDS_FILE_NAME
    if not records_path.exists():
        raise CrucibleRemoteError(
            f"Records file not found: {records_path}"
        )

    source_path = _find_source_file(dataset_dir)

    size_bytes = records_path.stat().st_size
    if source_path and source_path.exists():
        size_bytes += source_path.stat().st_size
    metadata = _build_metadata(dataset_name, size_bytes)
    if source_path:
        metadata["has_source"] = True

    safe_name = sanitize_remote_name(dataset_name)
    remote_dir = f"{_remote_datasets_dir(cluster)}/{safe_name}"
    session.mkdir_p(remote_dir)

    _upload_tar(session, remote_dir, records_path, metadata, source_path)

    return RemoteDatasetInfo(
        name=dataset_name,
        size_bytes=size_bytes,
        synced_at=metadata["synced_at"],
    )


SOURCE_DATA_FILE_NAME = "source.jsonl"


def _upload_tar(
    session: SshSession,
    remote_dir: str,
    records_path: Path,
    metadata: dict[str, Any],
    source_path: Path | None = None,
) -> None:
    """Create tar, upload, extract on remote, then clean up."""
    with tempfile.TemporaryDirectory() as tmp:
        meta_path = Path(tmp) / "metadata.json"
        meta_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
        )
        tar_path = Path(tmp) / "dataset.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(str(records_path), arcname=RECORDS_FILE_NAME)
            tar.add(str(meta_path), arcname="metadata.json")
            if source_path:
                tar.add(str(source_path), arcname=SOURCE_DATA_FILE_NAME)

        remote_tar = f"{remote_dir}/dataset.tar.gz"
        session.upload(tar_path, remote_tar)

    extract_cmd = (
        f"tar -xzf {remote_tar} -C {remote_dir} && rm -f {remote_tar}"
    )
    stdout, stderr, code = session.execute(extract_cmd)
    if code != 0:
        raise CrucibleRemoteError(
            f"Remote tar extraction failed (exit {code}): {stderr}"
        )


def list_remote_datasets(
    session: SshSession,
    cluster: ClusterConfig,
) -> list[RemoteDatasetInfo]:
    """List datasets present on the remote cluster."""
    datasets_dir = _remote_datasets_dir(cluster)
    stdout, _, _ = session.execute(
        f"ls -1 {datasets_dir} 2>/dev/null || true",
    )
    names = [n.strip() for n in stdout.strip().splitlines() if n.strip()]

    results: list[RemoteDatasetInfo] = []
    for name in names:
        info = _read_remote_metadata(session, datasets_dir, name)
        if info is not None:
            results.append(info)
    return results


def _read_remote_metadata(
    session: SshSession,
    datasets_dir: str,
    name: str,
) -> RemoteDatasetInfo | None:
    """Read and parse a single remote dataset's metadata.json."""
    meta_path = f"{datasets_dir}/{name}/metadata.json"
    stdout, _, code = session.execute(f"cat {meta_path}")
    if code != 0:
        return None
    try:
        data = json.loads(stdout)
        # Use actual remote directory size (includes source.jsonl + records)
        size_bytes = _get_remote_dir_size(
            session, f"{datasets_dir}/{name}",
        )
        if size_bytes == 0:
            size_bytes = data.get("size_bytes", 0)
        return RemoteDatasetInfo(
            name=str(data["name"]),
            size_bytes=int(size_bytes),
            synced_at=str(data["synced_at"]),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _get_remote_dir_size(session: SshSession, remote_dir: str) -> int:
    """Get total size in bytes of a remote directory via du."""
    stdout, _, code = session.execute(
        f"du -sb {remote_dir} 2>/dev/null | cut -f1",
    )
    if code == 0 and stdout.strip().isdigit():
        return int(stdout.strip())
    return 0


def delete_remote_dataset(
    session: SshSession,
    cluster: ClusterConfig,
    dataset_name: str,
) -> None:
    """Delete a dataset directory on the remote cluster."""
    remote_dir = f"{_remote_datasets_dir(cluster)}/{sanitize_remote_name(dataset_name)}"
    _, stderr, code = session.execute(f"rm -rf {remote_dir}")
    if code != 0:
        raise CrucibleRemoteError(
            f"Failed to delete remote dataset '{dataset_name}': {stderr}"
        )


def pull_remote_dataset(
    session: SshSession,
    cluster: ClusterConfig,
    dataset_name: str,
    data_root: Path,
) -> Path:
    """Download a remote dataset and register it locally.

    Downloads records.jsonl (and source.jsonl if present) into the
    local datasets directory and creates a manifest so the dataset
    appears in the registry.

    Returns the local dataset directory path.
    """
    datasets_dir = _remote_datasets_dir(cluster)
    safe_name = sanitize_remote_name(dataset_name)
    remote_dir = f"{datasets_dir}/{safe_name}"
    remote_records = f"{remote_dir}/{RECORDS_FILE_NAME}"

    _, _, code = session.execute(f"test -f {remote_records}")
    if code != 0:
        raise CrucibleRemoteError(
            f"Remote records not found: {remote_records}"
        )

    local_dir = data_root / DATASETS_DIR_NAME / dataset_name
    local_dir.mkdir(parents=True, exist_ok=True)

    # Download records
    session.download(remote_records, local_dir / RECORDS_FILE_NAME)

    # Download source data file if available
    remote_source = f"{remote_dir}/{SOURCE_DATA_FILE_NAME}"
    _, _, src_code = session.execute(
        f"test -f {remote_source}", timeout=10,
    )
    source_uri: str | None = None
    if src_code == 0:
        local_source = local_dir / SOURCE_DATA_FILE_NAME
        session.download(remote_source, local_source)
        source_uri = str(local_source)

    # Count records for manifest
    records_path = local_dir / RECORDS_FILE_NAME
    record_count = sum(
        1 for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

    # Write manifest so the dataset appears in the local registry
    from core.types import DatasetManifest
    from store.catalog_io import write_manifest_file

    manifest = DatasetManifest(
        dataset_name=dataset_name,
        created_at=datetime.now(timezone.utc),
        record_count=record_count,
        source_uri=source_uri,
    )
    write_manifest_file(local_dir, manifest, lance_written=False)

    return local_dir
