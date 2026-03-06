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
    CATALOG_FILE_NAME,
    DATASETS_DIR_NAME,
    MANIFEST_FILE_NAME,
    RECORDS_FILE_NAME,
    VERSIONS_DIR_NAME,
)
from core.errors import ForgeRemoteError
from core.slurm_types import ClusterConfig
from serve.ssh_connection import SshSession
from store.catalog_io import read_catalog_file


@dataclass(frozen=True)
class RemoteDatasetInfo:
    """Metadata for a dataset synced to a remote cluster."""

    name: str
    record_count: int
    version_id: str
    synced_at: str


def _remote_datasets_dir(cluster: ClusterConfig) -> str:
    """Return the remote datasets base directory."""
    return f"{cluster.remote_workspace}/{DATASETS_DIR_NAME}"


def _read_latest_version_id(catalog: dict[str, Any]) -> str:
    """Extract latest_version from a catalog dict."""
    latest = catalog.get("latest_version")
    if not latest or not isinstance(latest, str):
        raise ForgeRemoteError(
            "Dataset catalog has no latest_version entry."
        )
    return latest


def _count_records(version_dir: Path, records_path: Path) -> int:
    """Get record count from manifest.json or count lines."""
    manifest_path = version_dir / MANIFEST_FILE_NAME
    if manifest_path.exists():
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8"),
            )
            count = manifest.get("record_count")
            if isinstance(count, int) and count >= 0:
                return count
        except (json.JSONDecodeError, KeyError):
            pass
    # Fall back to counting lines in records file
    with records_path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def _build_metadata(
    name: str,
    record_count: int,
    version_id: str,
) -> dict[str, Any]:
    """Build metadata dict for the remote dataset."""
    return {
        "name": name,
        "record_count": record_count,
        "version_id": version_id,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }


def push_dataset(
    session: SshSession,
    cluster: ClusterConfig,
    dataset_name: str,
    data_root: Path,
) -> RemoteDatasetInfo:
    """Push a local dataset to the remote cluster.

    Reads the local catalog, tars the latest version's records
    with a metadata sidecar, uploads, and extracts on remote.
    """
    catalog_path = (
        data_root / DATASETS_DIR_NAME / dataset_name / CATALOG_FILE_NAME
    )
    catalog = read_catalog_file(catalog_path)
    version_id = _read_latest_version_id(catalog)

    version_dir = (
        data_root / DATASETS_DIR_NAME / dataset_name
        / VERSIONS_DIR_NAME / version_id
    )
    records_path = version_dir / RECORDS_FILE_NAME
    if not records_path.exists():
        raise ForgeRemoteError(
            f"Records file not found: {records_path}"
        )

    record_count = _count_records(version_dir, records_path)
    metadata = _build_metadata(dataset_name, record_count, version_id)

    remote_dir = f"{_remote_datasets_dir(cluster)}/{dataset_name}"
    session.mkdir_p(remote_dir)

    _upload_tar(session, remote_dir, records_path, metadata)

    return RemoteDatasetInfo(
        name=dataset_name,
        record_count=record_count,
        version_id=version_id,
        synced_at=metadata["synced_at"],
    )


def _upload_tar(
    session: SshSession,
    remote_dir: str,
    records_path: Path,
    metadata: dict[str, Any],
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

        remote_tar = f"{remote_dir}/dataset.tar.gz"
        session.upload(tar_path, remote_tar)

    extract_cmd = (
        f"tar -xzf {remote_tar} -C {remote_dir} && rm -f {remote_tar}"
    )
    stdout, stderr, code = session.execute(extract_cmd)
    if code != 0:
        raise ForgeRemoteError(
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
        return RemoteDatasetInfo(
            name=str(data["name"]),
            record_count=int(data["record_count"]),
            version_id=str(data["version_id"]),
            synced_at=str(data["synced_at"]),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def delete_remote_dataset(
    session: SshSession,
    cluster: ClusterConfig,
    dataset_name: str,
) -> None:
    """Delete a dataset directory on the remote cluster."""
    remote_dir = f"{_remote_datasets_dir(cluster)}/{dataset_name}"
    _, stderr, code = session.execute(f"rm -rf {remote_dir}")
    if code != 0:
        raise ForgeRemoteError(
            f"Failed to delete remote dataset '{dataset_name}': {stderr}"
        )


def pull_remote_dataset(
    session: SshSession,
    cluster: ClusterConfig,
    dataset_name: str,
    data_root: Path,
) -> Path:
    """Download a remote dataset's records to local cache.

    Returns the local path to the downloaded records.jsonl.
    """
    datasets_dir = _remote_datasets_dir(cluster)
    remote_records = (
        f"{datasets_dir}/{dataset_name}/{RECORDS_FILE_NAME}"
    )

    _, _, code = session.execute(f"test -f {remote_records}")
    if code != 0:
        raise ForgeRemoteError(
            f"Remote records not found: {remote_records}"
        )

    local_dir = data_root / "cache" / "remote-pulls" / dataset_name
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / RECORDS_FILE_NAME

    session.download(remote_records, local_path)
    return local_path
