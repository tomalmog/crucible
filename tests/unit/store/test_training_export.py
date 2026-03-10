"""Unit tests for training export helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.errors import CrucibleStoreError
from core.types import DataRecord, RecordMetadata, DatasetManifest, TrainingExportRequest
from store.training_export import export_training_shards


def _sample_record() -> DataRecord:
    metadata = RecordMetadata(
        source_uri="a.txt",
        language="en",
        quality_score=0.9,
        perplexity=1.2,
    )
    return DataRecord(record_id="rid", text="sample text", metadata=metadata)


def _sample_manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset_name="demo",
        created_at=datetime.now(timezone.utc),
        record_count=1,
    )


def test_export_training_shards_writes_manifest(tmp_path) -> None:
    """Export should create a training manifest file."""
    request = TrainingExportRequest(dataset_name="demo", output_dir=str(tmp_path), shard_size=1)

    manifest_path = export_training_shards(request, _sample_manifest(), [_sample_record()])

    assert manifest_path.exists()


def test_export_training_shards_rejects_invalid_shard_size(tmp_path) -> None:
    """Export should fail when shard size is not positive."""
    request = TrainingExportRequest(dataset_name="demo", output_dir=str(tmp_path), shard_size=0)

    with pytest.raises(CrucibleStoreError):
        export_training_shards(request, _sample_manifest(), [_sample_record()])

    assert True
