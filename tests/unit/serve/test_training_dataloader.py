"""Unit tests for training dataloader helpers."""

from __future__ import annotations

import pytest

from core.errors import CrucibleDependencyError
from core.types import DataLoaderOptions, DataRecord, RecordMetadata
from serve.training_dataloader import create_pytorch_dataloader, create_token_batches


def _build_records() -> list[DataRecord]:
    metadata = RecordMetadata(
        source_uri="tests/fixtures/raw/local_a.txt",
        language="en",
        quality_score=0.8,
        perplexity=2.0,
    )
    return [
        DataRecord(record_id="a", text="one two three", metadata=metadata),
        DataRecord(record_id="b", text="four five six", metadata=metadata),
    ]


def test_create_token_batches_batches_records() -> None:
    """Token batching should group records into configured batch size."""
    options = DataLoaderOptions(
        batch_size=1,
        shuffle=False,
        shuffle_buffer_size=2,
        max_token_length=10,
    )
    batches = create_token_batches(_build_records(), options, random_seed=7)

    assert len(batches) == 2


def test_create_pytorch_dataloader_raises_without_torch(monkeypatch) -> None:
    """DataLoader integration should fail clearly when torch is missing."""
    import builtins

    original_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    options = DataLoaderOptions(1, False, 2, 10)

    with pytest.raises(CrucibleDependencyError):
        create_pytorch_dataloader(_build_records(), options, random_seed=7)

    assert options.batch_size == 1
