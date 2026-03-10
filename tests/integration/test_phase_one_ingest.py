"""Integration tests for phase-one ingest workflow."""

from __future__ import annotations

from dataclasses import replace

from core.config import CrucibleConfig
from core.types import IngestOptions
from store.dataset_sdk import CrucibleClient


def test_phase_one_ingest_creates_dataset(tmp_path) -> None:
    """End-to-end flow should ingest and create a dataset with records."""
    config = replace(CrucibleConfig.from_env(), data_root=tmp_path)
    client = CrucibleClient(config)
    options = IngestOptions(
        dataset_name="integration-demo",
        source_uri="tests/fixtures/raw_valid",
    )

    dataset_name = client.ingest(options)

    assert dataset_name == "integration-demo"
    _, records = client.dataset("integration-demo").load_records()
    assert len(records) > 0
