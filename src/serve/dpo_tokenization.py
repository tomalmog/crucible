"""DPO tokenization for preference pair building.

This module tokenizes prompt+chosen and prompt+rejected sequences
separately, tracking prompt length for log-probability masking
during DPO loss computation.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.dpo_types import DpoExample
from serve.tokenization import VocabularyTokenizer


@dataclass(frozen=True)
class DpoTokenizedPair:
    """One tokenized DPO preference pair.

    Attributes:
        chosen_ids: Token ids for the full prompt+chosen sequence.
        rejected_ids: Token ids for the full prompt+rejected sequence.
        chosen_labels: Label ids for chosen (prompt masked with -100).
        rejected_labels: Label ids for rejected (prompt masked with -100).
        prompt_length: Number of prompt tokens for log-prob masking.
    """

    chosen_ids: tuple[int, ...]
    rejected_ids: tuple[int, ...]
    chosen_labels: tuple[int, ...]
    rejected_labels: tuple[int, ...]
    prompt_length: int


IGNORE_INDEX = -100


def build_dpo_pairs(
    examples: list[DpoExample],
    tokenizer: VocabularyTokenizer,
    max_length: int,
) -> list[DpoTokenizedPair]:
    """Tokenize DPO examples into preference pairs.

    Each example produces one pair with tokenized prompt+chosen
    and prompt+rejected sequences, plus a prompt_length for masking.

    Args:
        examples: Validated DPO preference examples.
        tokenizer: Fitted vocabulary tokenizer.
        max_length: Maximum sequence length after truncation.

    Returns:
        List of tokenized DPO pairs.
    """
    pairs: list[DpoTokenizedPair] = []
    for example in examples:
        pair = _build_single_pair(example, tokenizer, max_length)
        if pair is not None:
            pairs.append(pair)
    return pairs


def _build_single_pair(
    example: DpoExample,
    tokenizer: VocabularyTokenizer,
    max_length: int,
) -> DpoTokenizedPair | None:
    """Tokenize one DPO example into a preference pair."""
    prompt_ids = tokenizer.encode(example.prompt, max_length)
    chosen_ids = tokenizer.encode(example.chosen, max_length)
    rejected_ids = tokenizer.encode(example.rejected, max_length)
    full_chosen = prompt_ids + chosen_ids
    full_rejected = prompt_ids + rejected_ids
    if len(full_chosen) > max_length:
        full_chosen = full_chosen[:max_length]
    if len(full_rejected) > max_length:
        full_rejected = full_rejected[:max_length]
    if len(full_chosen) < 2 or len(full_rejected) < 2:
        return None
    prompt_length = min(len(prompt_ids), len(full_chosen), len(full_rejected))
    chosen_labels = _build_masked_labels(full_chosen, prompt_length)
    rejected_labels = _build_masked_labels(full_rejected, prompt_length)
    return DpoTokenizedPair(
        chosen_ids=tuple(full_chosen),
        rejected_ids=tuple(full_rejected),
        chosen_labels=tuple(chosen_labels),
        rejected_labels=tuple(rejected_labels),
        prompt_length=prompt_length,
    )


def _build_masked_labels(
    token_ids: list[int],
    prompt_length: int,
) -> list[int]:
    """Build labels with prompt positions masked to IGNORE_INDEX."""
    labels: list[int] = []
    for index, token_id in enumerate(token_ids):
        if index < prompt_length:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(token_id)
    return labels
