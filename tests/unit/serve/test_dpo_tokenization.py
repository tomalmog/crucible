"""Unit tests for DPO tokenization and preference pair building."""

from __future__ import annotations

from core.dpo_types import DpoExample
from serve.dpo_tokenization import (
    IGNORE_INDEX,
    DpoTokenizedPair,
    build_dpo_pairs,
)
from serve.tokenization import VocabularyTokenizer


def _build_tokenizer(texts: list[str]) -> VocabularyTokenizer:
    """Build a tokenizer fitted on the given texts."""
    tokenizer = VocabularyTokenizer.create()
    tokenizer.fit(texts)
    return tokenizer


def test_build_dpo_pairs_produces_correct_structure() -> None:
    """DPO pairs should have chosen/rejected ids and prompt length."""
    tokenizer = _build_tokenizer(["hello world good bad ugly"])
    examples = [
        DpoExample(prompt="hello world", chosen="good", rejected="bad ugly"),
    ]
    pairs = build_dpo_pairs(
        examples=examples,
        tokenizer=tokenizer,
        max_length=64,
    )

    assert len(pairs) == 1
    pair = pairs[0]
    assert len(pair.chosen_ids) > 0
    assert len(pair.rejected_ids) > 0
    assert pair.prompt_length > 0
    # Prompt positions should be masked in labels
    for i in range(pair.prompt_length):
        if i < len(pair.chosen_labels):
            assert pair.chosen_labels[i] == IGNORE_INDEX
        if i < len(pair.rejected_labels):
            assert pair.rejected_labels[i] == IGNORE_INDEX


def test_build_dpo_pairs_tracks_prompt_length() -> None:
    """Prompt length should match the tokenized prompt token count."""
    tokenizer = _build_tokenizer(["alpha beta gamma delta"])
    examples = [
        DpoExample(prompt="alpha beta", chosen="gamma", rejected="delta"),
    ]
    pairs = build_dpo_pairs(
        examples=examples,
        tokenizer=tokenizer,
        max_length=64,
    )

    assert len(pairs) == 1
    prompt_ids = tokenizer.encode("alpha beta", 64)
    assert pairs[0].prompt_length == len(prompt_ids)


def test_build_dpo_pairs_truncates_long_sequences() -> None:
    """Sequences exceeding max_length should be truncated."""
    tokenizer = _build_tokenizer(["a b c d e f g h i j k l"])
    examples = [
        DpoExample(prompt="a b c d e f", chosen="g h i", rejected="j k l"),
    ]
    max_length = 5
    pairs = build_dpo_pairs(
        examples=examples,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    assert len(pairs) == 1
    assert len(pairs[0].chosen_ids) <= max_length
    assert len(pairs[0].rejected_ids) <= max_length


def test_build_dpo_pairs_skips_too_short() -> None:
    """Pairs that would be shorter than 2 tokens should be skipped."""
    tokenizer = _build_tokenizer(["x"])
    examples = [
        DpoExample(prompt="x", chosen="x", rejected="x"),
    ]
    pairs = build_dpo_pairs(
        examples=examples,
        tokenizer=tokenizer,
        max_length=1,
    )

    # May produce 0 pairs since truncation to 1 token is too short
    assert len(pairs) == 0


def test_dpo_tokenized_pair_frozen() -> None:
    """DpoTokenizedPair should be immutable."""
    pair = DpoTokenizedPair(
        chosen_ids=(1, 2, 3),
        rejected_ids=(1, 2, 4),
        chosen_labels=(IGNORE_INDEX, 2, 3),
        rejected_labels=(IGNORE_INDEX, 2, 4),
        prompt_length=1,
    )
    try:
        pair.prompt_length = 5  # type: ignore[misc]
        assert False, "Expected FrozenInstanceError"
    except AttributeError:
        pass
