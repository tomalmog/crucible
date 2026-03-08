"""Unit tests for dataset store persistence."""

from __future__ import annotations

from dataclasses import replace

import pytest

from core.config import ForgeConfig
from core.errors import ForgeStoreError
from core.types import DataRecord, RecordMetadata, DatasetWriteRequest
from store.snapshot_store import DatasetStore


def _sample_record() -> DataRecord:
    metadata = RecordMetadata(
        source_uri="tests/fixtures/raw/local_a.txt",
        language="en",
        quality_score=0.9,
        perplexity=3.2,
    )
    return DataRecord(record_id="id-1", text="sample text", metadata=metadata)


def test_save_dataset_persists_manifest(tmp_path) -> None:
    """Store should create a dataset with manifest."""
    config = replace(ForgeConfig.from_env(), data_root=tmp_path)
    store = DatasetStore(config)
    request = DatasetWriteRequest(
        dataset_name="demo",
        records=(_sample_record(),),
    )

    manifest = store.save_dataset(request)

    assert manifest.dataset_name == "demo"


def test_load_records_returns_written_payload(tmp_path) -> None:
    """Store should return records written into a dataset."""
    config = replace(ForgeConfig.from_env(), data_root=tmp_path)
    store = DatasetStore(config)
    request = DatasetWriteRequest(
        dataset_name="demo",
        records=(_sample_record(),),
    )
    store.save_dataset(request)

    _, records = store.load_records("demo")

    assert records[0].text == "sample text"


def test_load_records_raises_for_unknown_dataset(tmp_path) -> None:
    """Loading should fail when dataset manifest is missing."""
    config = replace(ForgeConfig.from_env(), data_root=tmp_path)
    store = DatasetStore(config)

    with pytest.raises(ForgeStoreError):
        store.load_records("missing")

    assert (tmp_path / "datasets" / "missing").exists()
