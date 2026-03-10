"""Unit tests for dataset SDK training integration."""

from __future__ import annotations

from dataclasses import replace

import pytest

from core.config import CrucibleConfig
from core.errors import CrucibleServeError
from core.types import IngestOptions, TrainingOptions
from store.dataset_sdk import CrucibleClient


def test_dataset_train_raises_for_mismatched_dataset_name(tmp_path) -> None:
    """Dataset handle should reject training options for different dataset name."""
    config = replace(CrucibleConfig.from_env(), data_root=tmp_path)
    client = CrucibleClient(config)
    source_path = "tests/fixtures/raw_valid/local_a.txt"
    client.ingest(IngestOptions(dataset_name="demo", source_uri=source_path))
    dataset = client.dataset("demo")

    with pytest.raises(CrucibleServeError):
        dataset.train(TrainingOptions(dataset_name="other", output_dir=str(tmp_path / "out")))

    assert True
