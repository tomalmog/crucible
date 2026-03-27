"""Tokenization and sequence dataset utilities.

This module builds a lightweight vocabulary tokenizer and converts
text records into next-token prediction sequences for training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from core.types import DataRecord

if TYPE_CHECKING:
    from core.chat_types import ChatTokenizer


@dataclass
class VocabularyTokenizer:
    """Simple vocabulary tokenizer used by default training loop."""

    vocabulary: dict[str, int]

    @classmethod
    def create(cls) -> "VocabularyTokenizer":
        """Create tokenizer with special tokens.

        Returns:
            Tokenizer with pad and unknown tokens initialized.
        """
        return cls(vocabulary={"<pad>": 0, "<unk>": 1})

    def fit(self, texts: Iterable[str], max_vocabulary_size: int | None = None) -> None:
        """Fit tokenizer vocabulary from input texts.

        Args:
            texts: Input texts.
            max_vocabulary_size: Optional max token count including special tokens.
        """
        for text in texts:
            for token in _split_tokens(text):
                if token not in self.vocabulary:
                    if (
                        max_vocabulary_size is not None
                        and len(self.vocabulary) >= max_vocabulary_size
                    ):
                        return
                    self.vocabulary[token] = len(self.vocabulary)

    def encode(self, text: str, max_token_length: int) -> list[int]:
        """Encode text to token ids.

        Args:
            text: Input text.
            max_token_length: Maximum token count.

        Returns:
            Encoded token ids.
        """
        encoded = [self.vocabulary.get(token, 1) for token in _split_tokens(text)]
        if len(encoded) > max_token_length:
            return encoded[:max_token_length]
        return encoded

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids back into text.

        Args:
            token_ids: Token ids to decode.

        Returns:
            Whitespace-joined decoded text.
        """
        inverse_vocabulary = {value: key for key, value in self.vocabulary.items()}
        tokens = [
            inverse_vocabulary.get(token_id, "<unk>") for token_id in token_ids if token_id != 0
        ]
        return " ".join(tokens)


@dataclass(frozen=True)
class SequenceBatch:
    """One training batch containing input and target sequences."""

    inputs: list[list[int]]
    targets: list[list[int]]


def build_training_sequences(
    records: list[DataRecord],
    tokenizer: "ChatTokenizer",
    max_token_length: int,
) -> list[list[int]]:
    """Build token sequences from data records.

    Args:
        records: Dataset records.
        tokenizer: Fitted tokenizer.
        max_token_length: Maximum sequence length.

    Returns:
        List of token-id sequences.
    """
    sequences: list[list[int]] = []
    for record in records:
        token_ids = tokenizer.encode(record.text, max_token_length)
        if len(token_ids) < 2:
            continue
        sequences.append(token_ids)
    return sequences


def split_sequences(
    sequences: list[list[int]],
    validation_split: float,
) -> tuple[list[list[int]], list[list[int]]]:
    """Split sequences into training and validation sets.

    Args:
        sequences: Input sequences.
        validation_split: Fraction reserved for validation.

    Returns:
        Tuple of training and validation sequences.
    """
    if not sequences:
        return [], []
    validation_size = int(len(sequences) * validation_split)
    if validation_size < 1:
        validation_size = 1
    if validation_size >= len(sequences):
        validation_size = max(len(sequences) - 1, 0)
    split_index = len(sequences) - validation_size
    return sequences[:split_index], sequences[split_index:]


def build_sequence_batches(
    sequences: list[list[int]],
    batch_size: int,
) -> list[SequenceBatch]:
    """Create sequence batches with next-token targets.

    Args:
        sequences: Input token sequences.
        batch_size: Target batch size.

    Returns:
        Sequence batch list.
    """
    batches: list[SequenceBatch] = []
    batch_inputs: list[list[int]] = []
    batch_targets: list[list[int]] = []
    for sequence in sequences:
        batch_inputs.append(sequence[:-1])
        batch_targets.append(sequence[1:])
        if len(batch_inputs) >= batch_size:
            batches.append(SequenceBatch(inputs=batch_inputs, targets=batch_targets))
            batch_inputs, batch_targets = [], []
    if batch_inputs:
        batches.append(SequenceBatch(inputs=batch_inputs, targets=batch_targets))
    return batches


def collect_stop_token_ids(tokenizer: Any) -> set[int]:
    """Build the set of token IDs that should stop text generation.

    Always includes token 0 (pad).  For HuggingFace tokenizers, adds the
    model's EOS token and common chat end-of-turn tokens so generation
    stops cleanly instead of running to max_new_tokens.
    """
    stop: set[int] = {0}
    # HuggingFace tokenizers: AutoTokenizerAdapter wraps _tokenizer,
    # _HfTokenizerAdapter (eval) also wraps _tokenizer
    hf = getattr(tokenizer, "_tokenizer", None)
    if hf is not None:
        eos = getattr(hf, "eos_token_id", None)
        if isinstance(eos, int):
            stop.add(eos)
        # Some models (Qwen, ChatML, Llama 3, Phi) use additional end-of-turn tokens
        for special in ("<|im_end|>", "<|eot_id|>", "<|end|>"):
            try:
                tid = hf.convert_tokens_to_ids(special)
                if isinstance(tid, int) and tid != hf.unk_token_id:
                    stop.add(tid)
            except Exception:
                pass
    # Direct eos_token_id on the tokenizer wrapper itself (e.g. _HfTokenizerAdapter)
    direct_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(direct_eos, int) and direct_eos != 0:
        stop.add(direct_eos)
    return stop


def _split_tokens(text: str) -> list[str]:
    """Split text into lowercase whitespace tokens.

    Args:
        text: Input text.

    Returns:
        Token list.
    """
    return [token for token in text.lower().split() if token]
