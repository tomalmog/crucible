"""Unit tests for ingest checkpoint storage."""

from __future__ import annotations

from dataclasses import replace

import pytest

from core.config import CrucibleConfig
from core.errors import CrucibleIngestError
from core.types import DataRecord, RecordMetadata, SourceTextRecord
from ingest.checkpoint_store import IngestCheckpointStore


def _sample_data_record() -> DataRecord:
    metadata = RecordMetadata(
        source_uri="a.txt",
        language="en",
        quality_score=0.9,
        perplexity=1.0,
    )
    return DataRecord(record_id="id-1", text="sample", metadata=metadata)


def test_prepare_run_initializes_state(tmp_path) -> None:
    """New non-resume run should initialize checkpoint stage."""
    config = replace(CrucibleConfig.from_env(), data_root=tmp_path)
    checkpoint = IngestCheckpointStore(config.data_root, "demo")

    state = checkpoint.prepare_run("sig", resume=False)

    assert state.stage == "initialized"


def test_prepare_run_resume_requires_matching_signature(tmp_path) -> None:
    """Resume should fail when signature differs from checkpoint state."""
    config = replace(CrucibleConfig.from_env(), data_root=tmp_path)
    checkpoint = IngestCheckpointStore(config.data_root, "demo")
    checkpoint.prepare_run("sig-1", resume=False)

    with pytest.raises(CrucibleIngestError):
        checkpoint.prepare_run("sig-2", resume=True)

    assert True


def test_checkpoint_roundtrip_source_records(tmp_path) -> None:
    """Checkpoint should roundtrip source record payloads."""
    config = replace(CrucibleConfig.from_env(), data_root=tmp_path)
    checkpoint = IngestCheckpointStore(config.data_root, "demo")
    checkpoint.prepare_run("sig", resume=False)
    source_records = [SourceTextRecord(source_uri="a.txt", text="alpha")]
    checkpoint.save_source_records(source_records)

    loaded_records = checkpoint.load_source_records()

    assert loaded_records[0].text == "alpha"


def test_checkpoint_roundtrip_enriched_records(tmp_path) -> None:
    """Checkpoint should roundtrip enriched DataRecord payloads."""
    config = replace(CrucibleConfig.from_env(), data_root=tmp_path)
    checkpoint = IngestCheckpointStore(config.data_root, "demo")
    checkpoint.prepare_run("sig", resume=False)
    records = [_sample_data_record()]
    checkpoint.save_enriched_records(records)

    loaded_records = checkpoint.load_enriched_records()

    assert loaded_records[0].record_id == "id-1"
