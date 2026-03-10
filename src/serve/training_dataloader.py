"""PyTorch DataLoader integration for Crucible snapshots.

This module turns stored text records into tokenized training batches.
It provides deterministic shuffling and optional PyTorch DataLoader wiring.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, cast

from core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_TOKEN_LENGTH,
    DEFAULT_SHUFFLE_BUFFER_SIZE,
)
from core.errors import CrucibleDependencyError
from core.types import DataLoaderOptions, DataRecord


@dataclass
class WhitespaceTokenizer:
    """Simple incremental tokenizer for phase-one training streams."""

    vocabulary: dict[str, int]

    @classmethod
    def create(cls) -> "WhitespaceTokenizer":
        """Create tokenizer with reserved padding token."""
        return cls(vocabulary={"<pad>": 0})

    def encode(self, text: str, max_token_length: int) -> list[int]:
        """Encode text into integer token ids.

        Args:
            text: Input text.
            max_token_length: Maximum output token length.

        Returns:
            Token id list.
        """
        token_ids: list[int] = []
        for token in text.split():
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary)
            token_ids.append(self.vocabulary[token])
            if len(token_ids) >= max_token_length:
                break
        return token_ids


def build_default_dataloader_options() -> DataLoaderOptions:
    """Build default serving options for phase-one usage.

    Returns:
        DataLoaderOptions with conservative defaults.
    """
    return DataLoaderOptions(
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
        shuffle_buffer_size=DEFAULT_SHUFFLE_BUFFER_SIZE,
        max_token_length=DEFAULT_MAX_TOKEN_LENGTH,
    )


def create_token_batches(
    records: Iterable[DataRecord],
    options: DataLoaderOptions,
    random_seed: int,
) -> list[list[list[int]]]:
    """Create tokenized training batches from records.

    Args:
        records: Records to tokenize.
        options: Dataloader options.
        random_seed: Seed for deterministic shuffling.

    Returns:
        List of batches containing token-id sequences.
    """
    tokenizer = WhitespaceTokenizer.create()
    token_sequences = [
        tokenizer.encode(record.text, options.max_token_length) for record in records
    ]
    ordered_sequences = _shuffle_sequences(token_sequences, options, random_seed)
    return _batch_sequences(ordered_sequences, options.batch_size)


def create_pytorch_dataloader(
    records: Iterable[DataRecord],
    options: DataLoaderOptions,
    random_seed: int,
) -> Any:
    """Create a PyTorch DataLoader from snapshot records.

    Args:
        records: Records to stream.
        options: Dataloader behavior.
        random_seed: Shuffling seed.

    Returns:
        torch.utils.data.DataLoader instance.

    Raises:
        CrucibleDependencyError: If torch is unavailable.
    """
    try:
        import torch
    except ImportError as error:
        raise CrucibleDependencyError(
            "PyTorch DataLoader integration requires torch, but it is not installed. "
            "Install torch to use training streaming."
        ) from error
    batches = create_token_batches(records, options, random_seed)
    dataset = _TokenBatchDataset(batches)
    return torch.utils.data.DataLoader(cast(Any, dataset), batch_size=None)


class _TokenBatchDataset:
    """Indexable dataset wrapper for precomputed token batches."""

    def __init__(self, batches: list[list[list[int]]]) -> None:
        self._batches = batches

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, index: int) -> list[list[int]]:
        return self._batches[index]


def _shuffle_sequences(
    token_sequences: list[list[int]],
    options: DataLoaderOptions,
    random_seed: int,
) -> list[list[int]]:
    """Shuffle token sequences with deterministic seed.

    Args:
        token_sequences: Tokenized records.
        options: Shuffle options.
        random_seed: Deterministic seed.

    Returns:
        Ordered token sequences.
    """
    if not options.shuffle:
        return token_sequences
    randomizer = random.Random(random_seed)
    buffer_size = max(options.shuffle_buffer_size, 1)
    buffer: list[list[int]] = []
    shuffled: list[list[int]] = []
    for sequence in token_sequences:
        buffer.append(sequence)
        if len(buffer) >= buffer_size:
            randomizer.shuffle(buffer)
            shuffled.append(buffer.pop())
    randomizer.shuffle(buffer)
    shuffled.extend(buffer)
    return shuffled


def _batch_sequences(
    token_sequences: list[list[int]],
    batch_size: int,
) -> list[list[list[int]]]:
    """Batch token sequences.

    Args:
        token_sequences: Ordered token sequences.
        batch_size: Batch size.

    Returns:
        List of batches.
    """
    batches: list[list[list[int]]] = []
    current_batch: list[list[int]] = []
    for sequence in token_sequences:
        current_batch.append(sequence)
        if len(current_batch) == batch_size:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)
    return batches
