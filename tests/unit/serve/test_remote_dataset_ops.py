"""Unit tests for remote dataset operations over SSH."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.errors import CrucibleRemoteError
from core.slurm_types import ClusterConfig
from serve.remote_dataset_ops import (
    RemoteDatasetInfo,
    delete_remote_dataset,
    list_remote_datasets,
    pull_remote_dataset,
    push_dataset,
)


def _make_cluster() -> ClusterConfig:
    return ClusterConfig(
        name="test-hpc",
        host="hpc.example.com",
        user="jdoe",
        remote_workspace="/scratch/crucible",
    )


# -- push_dataset ----------------------------------------------------------


def test_push_dataset_raises_when_records_missing(tmp_path: Path) -> None:
    """CrucibleRemoteError is raised when records.jsonl does not exist."""
    session = MagicMock()
    cluster = _make_cluster()
    with pytest.raises(CrucibleRemoteError, match="Records file not found"):
        push_dataset(session, cluster, "my-ds", tmp_path)


@patch("serve.remote_dataset_ops._find_source_file", return_value=None)
@patch("serve.remote_dataset_ops._upload_tar")
def test_push_dataset_uploads_and_returns_info(
    mock_upload_tar: MagicMock,
    mock_find_source: MagicMock,
    tmp_path: Path,
) -> None:
    """push_dataset uploads the tar and returns a RemoteDatasetInfo."""
    dataset_dir = tmp_path / "datasets" / "my-ds"
    dataset_dir.mkdir(parents=True)
    records = dataset_dir / "records.jsonl"
    records.write_text('{"text": "hello"}\n')

    session = MagicMock()
    cluster = _make_cluster()
    result = push_dataset(session, cluster, "my-ds", tmp_path)

    assert isinstance(result, RemoteDatasetInfo)
    assert result.name == "my-ds"
    assert result.size_bytes == records.stat().st_size
    session.mkdir_p.assert_called_once()
    mock_upload_tar.assert_called_once()


# -- list_remote_datasets --------------------------------------------------


def test_list_remote_datasets_returns_info_list() -> None:
    """Two valid datasets in ls output produce a list of length 2."""
    meta1 = json.dumps({"name": "ds1", "size_bytes": 100, "synced_at": "t1"})
    meta2 = json.dumps({"name": "ds2", "size_bytes": 200, "synced_at": "t2"})
    session = MagicMock()
    session.execute.side_effect = [
        ("ds1\nds2\n", "", 0),
        (meta1, "", 0),            # cat metadata.json ds1
        ("12345\n", "", 0),         # du -sb ds1
        (meta2, "", 0),            # cat metadata.json ds2
        ("67890\n", "", 0),         # du -sb ds2
    ]
    cluster = _make_cluster()
    results = list_remote_datasets(session, cluster)
    assert len(results) == 2


def test_list_remote_datasets_skips_invalid_metadata() -> None:
    """Datasets with unparseable metadata are silently skipped."""
    session = MagicMock()
    session.execute.side_effect = [
        ("bad-ds\n", "", 0),
        ("not-valid-json{{", "", 0),
    ]
    cluster = _make_cluster()
    results = list_remote_datasets(session, cluster)
    assert len(results) == 0


def test_list_remote_datasets_empty_returns_empty() -> None:
    """Empty ls output returns an empty list."""
    session = MagicMock()
    session.execute.return_value = ("", "", 0)
    cluster = _make_cluster()
    results = list_remote_datasets(session, cluster)
    assert results == []


# -- delete_remote_dataset --------------------------------------------------


def test_delete_remote_dataset_executes_rm() -> None:
    """execute is called with rm -rf targeting the dataset directory."""
    session = MagicMock()
    session.execute.return_value = ("", "", 0)
    cluster = _make_cluster()
    delete_remote_dataset(session, cluster, "old-ds")
    cmd = session.execute.call_args.args[0]
    assert "rm -rf" in cmd
    assert "old-ds" in cmd


def test_delete_remote_dataset_raises_on_failure() -> None:
    """Non-zero exit code raises CrucibleRemoteError."""
    session = MagicMock()
    session.execute.return_value = ("", "permission denied", 1)
    cluster = _make_cluster()
    with pytest.raises(CrucibleRemoteError, match="Failed to delete"):
        delete_remote_dataset(session, cluster, "old-ds")


# -- pull_remote_dataset ----------------------------------------------------


def test_pull_remote_dataset_downloads_and_registers(tmp_path: Path) -> None:
    """Pull downloads records, creates manifest, returns dataset dir."""
    session = MagicMock()
    session.execute.side_effect = [
        ("", "", 0),   # test -f records.jsonl
        ("", "", 1),   # test -f source.jsonl (not present)
    ]
    # Simulate download writing the records file
    def fake_download(remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text('{"record_id":"1","text":"hello"}\n')
    session.download.side_effect = fake_download
    cluster = _make_cluster()
    result = pull_remote_dataset(session, cluster, "pull-ds", tmp_path)
    assert result == tmp_path / "datasets" / "pull-ds"
    assert (result / "manifest.json").exists()
    assert (result / "records.jsonl").exists()


def test_pull_remote_dataset_raises_when_not_found(tmp_path: Path) -> None:
    """CrucibleRemoteError is raised when remote records file does not exist."""
    session = MagicMock()
    session.execute.return_value = ("", "", 1)
    cluster = _make_cluster()
    with pytest.raises(CrucibleRemoteError, match="Remote records not found"):
        pull_remote_dataset(session, cluster, "missing-ds", tmp_path)
