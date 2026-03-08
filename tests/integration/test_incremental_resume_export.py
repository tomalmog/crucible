"""Integration tests for resume and training export workflows."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from core.config import ForgeConfig
from core.errors import ForgeStoreError
from core.types import IngestOptions
from ingest.pipeline import ingest_dataset
from store.dataset_sdk import ForgeClient
from store.snapshot_store import DatasetStore


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_resume_uses_checkpoint_without_source_files(tmp_path: Path, monkeypatch) -> None:
    """Resume should finish from checkpoint after a save failure."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_file = source_dir / "a.txt"
    _write_text(source_file, "alpha")
    config = replace(ForgeConfig.from_env(), data_root=tmp_path / "forge")
    options = IngestOptions(dataset_name="demo", source_uri=str(source_dir))

    original_save_dataset = DatasetStore.save_dataset

    def _failing_save_dataset(self, request):
        raise ForgeStoreError("forced failure")

    monkeypatch.setattr(DatasetStore, "save_dataset", _failing_save_dataset)
    with pytest.raises(ForgeStoreError):
        ingest_dataset(options, config)
    source_file.unlink()
    monkeypatch.setattr(DatasetStore, "save_dataset", original_save_dataset)

    dataset_name = ingest_dataset(replace(options, resume=True), config)

    assert dataset_name == "demo"


def test_export_training_creates_manifest(tmp_path: Path) -> None:
    """Training export should create manifest file on disk."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_text(source_dir / "a.txt", "alpha training sample")
    config = replace(ForgeConfig.from_env(), data_root=tmp_path / "forge")
    client = ForgeClient(config)
    client.ingest(IngestOptions(dataset_name="demo", source_uri=str(source_dir)))

    manifest_path = client.dataset("demo").export_training(output_dir=str(tmp_path / "exports"))

    assert Path(manifest_path).exists()
