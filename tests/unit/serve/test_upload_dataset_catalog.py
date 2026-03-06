"""Tests for _upload_dataset_catalog in remote_job_submitter.

Verifies the full round-trip: create a local dataset, call the upload
function with a fake SSH session that writes to a local "remote" dir,
then confirm a ForgeClient pointing at that dir can load the records.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.config import ForgeConfig
from core.constants import (
    CATALOG_FILE_NAME,
    DATASETS_DIR_NAME,
    MANIFEST_FILE_NAME,
    RECORDS_FILE_NAME,
    VERSIONS_DIR_NAME,
)
from core.types import DataRecord, RecordMetadata, SnapshotManifest
from store.catalog_io import update_catalog, write_manifest_file
from store.record_payload import write_data_records_jsonl


def _make_records(count: int = 5) -> list[DataRecord]:
    """Create dummy DataRecords."""
    return [
        DataRecord(
            record_id=f"rec-{i:04d}",
            text=f"Sample text for record {i}.",
            metadata=RecordMetadata(
                source_uri="test.jsonl",
                language="en",
                quality_score=0.9,
                perplexity=15.0,
                extra_fields={},
            ),
        )
        for i in range(count)
    ]


def _create_local_dataset(
    data_root: Path, name: str, records: list[DataRecord],
) -> str:
    """Create a minimal dataset in the local .forge structure.

    Returns the version_id.
    """
    from datetime import datetime, timezone

    ds_dir = data_root / DATASETS_DIR_NAME / name
    version_id = f"{name}-20260301T120000000000Z-abcdef1234"

    version_dir = ds_dir / VERSIONS_DIR_NAME / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    # Write records.jsonl
    write_data_records_jsonl(version_dir / RECORDS_FILE_NAME, records)

    # Write manifest.json
    manifest = SnapshotManifest(
        dataset_name=name,
        version_id=version_id,
        created_at=datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc),
        parent_version=None,
        recipe_steps=(),
        record_count=len(records),
        source_uri="test.jsonl",
    )
    write_manifest_file(version_dir, manifest, lance_written=False)

    # Write catalog.json
    catalog_path = ds_dir / CATALOG_FILE_NAME
    update_catalog(catalog_path, manifest)

    return version_id


class FakeSshSession:
    """SSH session stand-in that writes to a local 'remote' directory.

    Tracks mkdir_p calls and intercepts upload + execute to simulate
    uploading a tarball and extracting it on a remote host.
    """

    def __init__(self, remote_root: Path) -> None:
        self._remote_root = remote_root
        self.mkdirs: list[str] = []
        self.uploads: list[tuple[Path, str]] = []
        self.commands: list[str] = []

    def _local(self, remote_path: str) -> Path:
        """Map a remote absolute path to a local directory."""
        # remote paths look like /workspace/job-id/.forge/datasets/...
        # strip leading /
        rel = remote_path.lstrip("/")
        return self._remote_root / rel

    def mkdir_p(self, remote_path: str) -> None:
        self.mkdirs.append(remote_path)
        self._local(remote_path).mkdir(parents=True, exist_ok=True)

    def upload(self, local_path: Path, remote_path: str) -> None:
        self.uploads.append((local_path, remote_path))
        dest = self._local(remote_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(str(local_path), str(dest))

    def execute(self, command: str, timeout: int = 60) -> tuple[str, str, int]:
        self.commands.append(command)
        # Handle tar extraction commands
        if "tar xzf" in command:
            # Parse: "cd /some/dir && tar xzf file.tar.gz && rm file.tar.gz"
            parts = command.split("&&")
            cd_part = parts[0].strip()
            tar_dir = cd_part.replace("cd ", "").strip()
            local_dir = self._local(tar_dir)

            # Find the tar.gz file
            tar_part = parts[1].strip()
            tar_name = tar_part.split()[-1]
            tar_path = local_dir / tar_name

            if tar_path.exists():
                with tarfile.open(str(tar_path), "r:gz") as tar:
                    tar.extractall(path=str(local_dir), filter="data")
                tar_path.unlink()

            return "", "", 0
        return "", "", 0


@pytest.fixture()
def local_forge(tmp_path: Path) -> Path:
    """Create a .forge data root with a test dataset."""
    return tmp_path / "local_forge"


@pytest.fixture()
def remote_root(tmp_path: Path) -> Path:
    """Root directory simulating the remote filesystem."""
    return tmp_path / "remote_fs"


def test_upload_dataset_catalog_creates_valid_remote_structure(
    local_forge: Path, remote_root: Path,
) -> None:
    """Uploaded catalog + version allows a ForgeClient to load records."""
    from serve.remote_data_upload import _upload_dataset_catalog

    # --- Set up local dataset ---
    records = _make_records(10)
    version_id = _create_local_dataset(local_forge, "demo", records)

    # --- Simulate upload ---
    workdir = "/workspace/job-123"
    session = FakeSshSession(remote_root)
    _upload_dataset_catalog(session, local_forge, "demo", workdir)

    # --- Verify remote directory structure ---
    remote_ds = remote_root / "workspace/job-123/.forge/datasets/demo"
    assert (remote_ds / CATALOG_FILE_NAME).exists()
    assert (remote_ds / VERSIONS_DIR_NAME / version_id / MANIFEST_FILE_NAME).exists()
    assert (remote_ds / VERSIONS_DIR_NAME / version_id / RECORDS_FILE_NAME).exists()

    # --- Verify catalog content is minimal (single version) ---
    catalog = json.loads((remote_ds / CATALOG_FILE_NAME).read_text())
    assert catalog["latest_version"] == version_id
    assert len(catalog["versions"]) == 1
    assert catalog["versions"][0]["version_id"] == version_id
    assert catalog["versions"][0]["record_count"] == 10


def test_upload_dataset_catalog_records_round_trip(
    local_forge: Path, remote_root: Path,
) -> None:
    """ForgeClient can load records from the uploaded remote structure."""
    from serve.remote_data_upload import _upload_dataset_catalog
    from store.dataset_sdk import ForgeClient

    records = _make_records(7)
    _create_local_dataset(local_forge, "mydata", records)

    workdir = "/workspace/job-456"
    session = FakeSshSession(remote_root)
    _upload_dataset_catalog(session, local_forge, "mydata", workdir)

    # Point a ForgeClient at the simulated remote .forge
    remote_forge = remote_root / "workspace/job-456/.forge"
    config = ForgeConfig(
        data_root=remote_forge,
        s3_region=None,
        s3_profile=None,
        random_seed=42,
    )
    client = ForgeClient(config)

    # Load records — this is exactly what the remote agent does
    manifest, loaded = client.dataset("mydata").load_records()
    assert manifest.version_id.startswith("mydata-")
    assert len(loaded) == 7
    assert loaded[0].record_id == "rec-0000"
    assert loaded[0].text == "Sample text for record 0."


def test_upload_dataset_catalog_skips_lance(
    local_forge: Path, remote_root: Path,
) -> None:
    """Lance directory is NOT uploaded (remote only needs JSONL)."""
    from serve.remote_data_upload import _upload_dataset_catalog

    records = _make_records(3)
    version_id = _create_local_dataset(local_forge, "ds", records)

    # Create a fake data.lance directory locally
    lance_dir = (
        local_forge / DATASETS_DIR_NAME / "ds"
        / VERSIONS_DIR_NAME / version_id / "data.lance"
    )
    lance_dir.mkdir()
    (lance_dir / "some_file.lance").write_text("fake lance data")

    workdir = "/workspace/job-789"
    session = FakeSshSession(remote_root)
    _upload_dataset_catalog(session, local_forge, "ds", workdir)

    remote_version = (
        remote_root / "workspace/job-789/.forge/datasets/ds"
        / VERSIONS_DIR_NAME / version_id
    )
    assert not (remote_version / "data.lance").exists()
    assert (remote_version / RECORDS_FILE_NAME).exists()


def test_upload_dataset_catalog_missing_catalog_raises(
    local_forge: Path, remote_root: Path,
) -> None:
    """Raises ForgeRemoteError when catalog.json does not exist."""
    from core.errors import ForgeStoreError
    from serve.remote_data_upload import _upload_dataset_catalog

    session = FakeSshSession(remote_root)
    with pytest.raises(ForgeStoreError, match="catalog not found"):
        _upload_dataset_catalog(session, local_forge, "nonexistent", "/w")


def test_upload_dataset_catalog_empty_latest_raises(
    local_forge: Path, remote_root: Path,
) -> None:
    """Raises ForgeRemoteError when latest_version is empty."""
    from core.errors import ForgeRemoteError
    from serve.remote_data_upload import _upload_dataset_catalog

    ds_dir = local_forge / DATASETS_DIR_NAME / "bad"
    ds_dir.mkdir(parents=True)
    (ds_dir / CATALOG_FILE_NAME).write_text(
        json.dumps({"latest_version": None, "versions": []})
    )

    session = FakeSshSession(remote_root)
    with pytest.raises(ForgeRemoteError, match="no latest version"):
        _upload_dataset_catalog(session, local_forge, "bad", "/w")


def test_handle_data_strategy_dispatches_for_record_methods(
    local_forge: Path, remote_root: Path,
) -> None:
    """_handle_data_strategy calls _upload_dataset_catalog for train method."""
    from serve.remote_data_upload import _handle_data_strategy

    records = _make_records(4)
    _create_local_dataset(local_forge, "strat-ds", records)

    workdir = "/workspace/job-strat"
    session = FakeSshSession(remote_root)
    method_args: dict[str, object] = {"dataset_name": "strat-ds"}

    _handle_data_strategy(
        session, "shared", "", method_args, workdir,
        training_method="train", data_root=local_forge,
    )

    # Catalog should exist on remote
    remote_ds = remote_root / "workspace/job-strat/.forge/datasets/strat-ds"
    assert (remote_ds / CATALOG_FILE_NAME).exists()


def test_handle_data_strategy_skips_non_record_methods(
    local_forge: Path, remote_root: Path,
) -> None:
    """_handle_data_strategy does NOT upload catalog for SFT (non-record method)."""
    from serve.remote_data_upload import _handle_data_strategy

    records = _make_records(2)
    _create_local_dataset(local_forge, "sft-ds", records)

    workdir = "/workspace/job-sft"
    session = FakeSshSession(remote_root)
    method_args: dict[str, object] = {"dataset_name": "sft-ds"}

    _handle_data_strategy(
        session, "shared", "", method_args, workdir,
        training_method="sft", data_root=local_forge,
    )

    # No catalog should be uploaded for SFT
    remote_ds = remote_root / "workspace/job-sft/.forge/datasets/sft-ds"
    assert not remote_ds.exists()


def test_handle_data_strategy_falls_through_without_dataset_name(
    local_forge: Path, remote_root: Path,
) -> None:
    """Without dataset_name, falls through to existing scp/shared logic."""
    from serve.remote_data_upload import _handle_data_strategy

    workdir = "/workspace/job-no-ds"
    session = FakeSshSession(remote_root)
    method_args: dict[str, object] = {}

    _handle_data_strategy(
        session, "shared", "", method_args, workdir,
        training_method="train", data_root=local_forge,
    )

    # No catalog upload happened — no datasets dir created
    remote_forge = remote_root / "workspace/job-no-ds/.forge"
    assert not remote_forge.exists()
