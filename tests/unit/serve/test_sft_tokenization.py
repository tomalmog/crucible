"""Unit tests for SFT tokenization and sequence building."""

from __future__ import annotations

from core.sft_types import SftExample
from serve.sft_tokenization import (
    IGNORE_INDEX,
    SftSequence,
    build_sft_sequences,
    pack_sft_sequences,
)
from serve.tokenization import VocabularyTokenizer


def _build_tokenizer(texts: list[str]) -> VocabularyTokenizer:
    """Build a tokenizer fitted on the given texts."""
    tokenizer = VocabularyTokenizer.create()
    tokenizer.fit(texts)
    return tokenizer


def test_build_sft_sequences_masks_prompt_tokens() -> None:
    """Prompt token positions should receive IGNORE_INDEX labels when masking is enabled."""
    tokenizer = _build_tokenizer(["hello world goodbye moon"])
    examples = [
        SftExample(prompt="hello world", response="goodbye moon"),
    ]
    sequences = build_sft_sequences(
        examples=examples,
        tokenizer=tokenizer,
        max_token_length=64,
        mask_prompt_tokens=True,
    )

    assert len(sequences) == 1
    seq = sequences[0]
    prompt_ids = tokenizer.encode("hello world", 64)
    prompt_length = len(prompt_ids)
    # Prompt positions should be masked
    for i in range(prompt_length):
        assert seq.labels[i] == IGNORE_INDEX
    # Response positions should have actual token ids
    for i in range(prompt_length, len(seq.labels)):
        assert seq.labels[i] != IGNORE_INDEX


def test_build_sft_sequences_without_masking() -> None:
    """All positions should have actual labels when masking is disabled."""
    tokenizer = _build_tokenizer(["hello world goodbye moon"])
    examples = [
        SftExample(prompt="hello world", response="goodbye moon"),
    ]
    sequences = build_sft_sequences(
        examples=examples,
        tokenizer=tokenizer,
        max_token_length=64,
        mask_prompt_tokens=False,
    )

    assert len(sequences) == 1
    seq = sequences[0]
    for label in seq.labels:
        assert label != IGNORE_INDEX


def test_build_sft_sequences_truncates_long() -> None:
    """Sequences exceeding max_token_length should be truncated."""
    tokenizer = _build_tokenizer(["a b c d e f g h i j k l"])
    examples = [
        SftExample(prompt="a b c d e f", response="g h i j k l"),
    ]
    max_length = 6
    sequences = build_sft_sequences(
        examples=examples,
        tokenizer=tokenizer,
        max_token_length=max_length,
        mask_prompt_tokens=True,
    )

    assert len(sequences) == 1
    assert len(sequences[0].input_ids) <= max_length
    assert len(sequences[0].labels) <= max_length


def test_pack_sft_sequences_combines_short() -> None:
    """Packing should combine short sequences into longer ones."""
    seq1 = SftSequence(input_ids=(1, 2), labels=(IGNORE_INDEX, 2))
    seq2 = SftSequence(input_ids=(3, 4), labels=(IGNORE_INDEX, 4))
    seq3 = SftSequence(input_ids=(5, 6), labels=(5, 6))

    packed = pack_sft_sequences([seq1, seq2, seq3], max_token_length=6)

    assert len(packed) == 1
    assert packed[0].input_ids == (1, 2, 3, 4, 5, 6)
    assert packed[0].labels == (IGNORE_INDEX, 2, IGNORE_INDEX, 4, 5, 6)


def test_sft_sequence_frozen() -> None:
    """SftSequence should be immutable."""
    seq = SftSequence(input_ids=(1, 2, 3), labels=(IGNORE_INDEX, 2, 3))
    try:
        seq.input_ids = (4, 5, 6)  # type: ignore[misc]
        assert False, "Expected FrozenInstanceError"
    except AttributeError:
        pass
