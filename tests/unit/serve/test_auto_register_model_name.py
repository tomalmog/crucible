"""Tests that auto_register_remote_model uses record.model_name.

This is the actual feature: when a user types "My-Transformer" in the
UI and submits to a remote cluster, the model must be registered under
"My-Transformer" — not an auto-generated name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.slurm_types import ClusterConfig, RemoteJobRecord
from store.cluster_registry import save_cluster
from store.model_registry import ModelRegistry
from store.remote_job_store import now_iso


def _setup_cluster(data_root: Path) -> None:
    """Save a dummy cluster so auto_register_remote_model can load it."""
    cluster = ClusterConfig(
        name="test-hpc",
        host="hpc.example.com",
        user="jdoe",
    )
    save_cluster(data_root, cluster)


def _make_record(
    model_name: str = "",
    job_id: str = "rj-abc123def456",
) -> RemoteJobRecord:
    ts = now_iso()
    return RemoteJobRecord(
        job_id=job_id,
        slurm_job_id="12345",
        cluster_name="test-hpc",
        training_method="train",
        state="running",
        submitted_at=ts,
        updated_at=ts,
        remote_output_dir="/scratch/crucible/rj-abc123def456",
        model_name=model_name,
    )


def test_auto_register_uses_user_provided_name(tmp_path: Path) -> None:
    """When record.model_name is set, the model is registered under that name."""
    from serve.remote_model_registry import auto_register_remote_model

    _setup_cluster(tmp_path)
    record = _make_record(model_name="My-Transformer")

    auto_register_remote_model(
        tmp_path, record, "/remote/path/model.pt",
    )

    registry = ModelRegistry(tmp_path)
    names = registry.list_model_names()
    assert "My-Transformer" in names


def test_auto_register_uses_fallback_when_no_name(tmp_path: Path) -> None:
    """When record.model_name is empty, falls back to auto-generated name."""
    from serve.remote_model_registry import auto_register_remote_model

    _setup_cluster(tmp_path)
    record = _make_record(model_name="")

    auto_register_remote_model(
        tmp_path, record, "/remote/path/model.pt",
    )

    registry = ModelRegistry(tmp_path)
    names = registry.list_model_names()
    expected = f"remote-train-{record.job_id[:16]}"
    assert expected in names


def test_auto_register_model_has_correct_remote_path(tmp_path: Path) -> None:
    """Registered model entry points to the correct remote path and host."""
    from serve.remote_model_registry import auto_register_remote_model

    _setup_cluster(tmp_path)
    record = _make_record(model_name="Cool-Model")

    auto_register_remote_model(
        tmp_path, record, "/remote/output/model.pt",
    )

    registry = ModelRegistry(tmp_path)
    entry = registry.get_model("Cool-Model")
    assert entry.model_name == "Cool-Model"
    assert entry.remote_path == "/remote/output/model.pt"
    assert entry.remote_host == "hpc.example.com"
    assert entry.run_id == record.job_id
    assert entry.location_type == "remote"


def test_pull_remote_model_uses_stored_name(tmp_path: Path) -> None:
    """pull_remote_model should use record.model_name when no explicit name given."""
    from serve.remote_model_puller import pull_remote_model
    from store.remote_job_store import save_remote_job
    from unittest.mock import MagicMock, patch
    import tarfile

    _setup_cluster(tmp_path)
    record = _make_record(model_name="User-Named-Model")
    from dataclasses import replace
    record = replace(
        record,
        state="completed",
        model_path_remote="/remote/output/model.pt",
    )
    save_remote_job(tmp_path, record)

    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_session.execute.side_effect = [
        ("", "", 0),              # test -d
        ("100M\t/dir", "", 0),   # du -sh
        ("", "", 0),              # tar czf
        ("50M\t/file", "", 0),   # du -sh compressed
        ("", "", 0),              # rm -f
    ]

    staging = tmp_path / "staging"
    staging.mkdir()
    model_content_dir = staging / "model_files"
    model_content_dir.mkdir()
    (model_content_dir / "model.pt").write_bytes(b"fake model")
    tar_path = staging / "model_download.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(str(model_content_dir / "model.pt"), arcname="model.pt")

    def fake_download(remote_path: str, local_path: Path) -> None:
        import shutil
        shutil.copy2(str(tar_path), str(local_path))

    mock_session.download.side_effect = fake_download

    with patch("serve.remote_model_puller.SshSession", return_value=mock_session):
        result = pull_remote_model(tmp_path, record.job_id)

    registry = ModelRegistry(tmp_path)
    names = registry.list_model_names()
    assert "User-Named-Model" in names
