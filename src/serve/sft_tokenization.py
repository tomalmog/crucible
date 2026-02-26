"""SFT sequence building with prompt masking.

This module tokenizes prompt/response pairs and builds label masks that
ignore prompt tokens during loss computation, ensuring the model only
learns to generate responses.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.chat_types import ChatTokenizer
from core.sft_types import SftExample

IGNORE_INDEX = -100


@dataclass(frozen=True)
class SftSequence:
    """One tokenized SFT training sequence with label mask.

    Attributes:
        input_ids: Token ids for the full prompt+response sequence.
        labels: Target token ids with IGNORE_INDEX for masked positions.
    """

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


def build_sft_sequences(
    examples: list[SftExample],
    tokenizer: ChatTokenizer,
    max_token_length: int,
    mask_prompt_tokens: bool,
) -> list[SftSequence]:
    """Tokenize SFT examples into masked training sequences.

    Each example is tokenized as prompt+response, then labels are built
    so that prompt positions are masked with IGNORE_INDEX when masking
    is enabled.

    Args:
        examples: Validated SFT prompt/response pairs.
        tokenizer: Fitted vocabulary tokenizer.
        max_token_length: Maximum sequence length after truncation.
        mask_prompt_tokens: Whether to mask prompt token labels.

    Returns:
        List of tokenized SFT sequences with label masks.
    """
    sequences: list[SftSequence] = []
    for example in examples:
        sequence = _build_single_sequence(
            example, tokenizer, max_token_length, mask_prompt_tokens
        )
        if sequence is not None:
            sequences.append(sequence)
    return sequences


def pack_sft_sequences(
    sequences: list[SftSequence],
    max_token_length: int,
) -> list[SftSequence]:
    """Pack multiple short sequences into longer sequences for efficiency.

    Concatenates consecutive sequences up to max_token_length, preserving
    label masks across boundaries.

    Args:
        sequences: Input SFT sequences to pack.
        max_token_length: Maximum packed sequence length.

    Returns:
        Packed sequences, potentially fewer but longer than input.
    """
    if not sequences:
        return []
    packed: list[SftSequence] = []
    current_ids: list[int] = []
    current_labels: list[int] = []
    for sequence in sequences:
        candidate_ids = list(sequence.input_ids)
        candidate_labels = list(sequence.labels)
        combined_length = len(current_ids) + len(candidate_ids)
        if combined_length <= max_token_length:
            current_ids.extend(candidate_ids)
            current_labels.extend(candidate_labels)
        else:
            if current_ids:
                packed.append(_finalize_packed(current_ids, current_labels))
            if len(candidate_ids) <= max_token_length:
                current_ids = candidate_ids
                current_labels = candidate_labels
            else:
                packed.append(_finalize_packed(
                    candidate_ids[:max_token_length],
                    candidate_labels[:max_token_length],
                ))
                current_ids = []
                current_labels = []
    if current_ids:
        packed.append(_finalize_packed(current_ids, current_labels))
    return packed


def _build_single_sequence(
    example: SftExample,
    tokenizer: ChatTokenizer,
    max_token_length: int,
    mask_prompt_tokens: bool,
) -> SftSequence | None:
    """Tokenize one SFT example into a masked sequence."""
    full_prompt = example.prompt
    if example.system_prompt:
        full_prompt = example.system_prompt + " " + example.prompt
    prompt_ids = tokenizer.encode(full_prompt, max_token_length)
    response_ids = tokenizer.encode(example.response, max_token_length)
    combined_ids = prompt_ids + response_ids
    if len(combined_ids) > max_token_length:
        combined_ids = combined_ids[:max_token_length]
    if len(combined_ids) < 2:
        return None
    prompt_length = min(len(prompt_ids), len(combined_ids))
    labels = _build_labels(combined_ids, prompt_length, mask_prompt_tokens)
    return SftSequence(
        input_ids=tuple(combined_ids),
        labels=tuple(labels),
    )


def _build_labels(
    token_ids: list[int],
    prompt_length: int,
    mask_prompt_tokens: bool,
) -> list[int]:
    """Build label list with optional prompt masking."""
    labels: list[int] = []
    for index, token_id in enumerate(token_ids):
        if mask_prompt_tokens and index < prompt_length:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(token_id)
    return labels


def _finalize_packed(
    ids: list[int],
    labels: list[int],
) -> SftSequence:
    """Create SftSequence from accumulated packed buffers."""
    return SftSequence(
        input_ids=tuple(ids),
        labels=tuple(labels),
    )
