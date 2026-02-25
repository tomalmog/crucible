"""Typed models for chat inference workflows.

This module defines explicit request/response contracts used by
CLI, SDK, and serving layers for model chat interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from core.constants import (
    DEFAULT_CHAT_MAX_NEW_TOKENS,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_CHAT_TOP_K,
    DEFAULT_MAX_TOKEN_LENGTH,
    DEFAULT_POSITION_EMBEDDING_TYPE,
    DEFAULT_TRAIN_ATTENTION_HEADS,
    DEFAULT_TRAIN_DROPOUT,
    DEFAULT_TRAIN_HIDDEN_DIM,
    DEFAULT_TRAIN_MLP_HIDDEN_DIM,
    DEFAULT_TRAIN_MLP_LAYERS,
    DEFAULT_TRAIN_NUM_LAYERS,
)
from core.types import PositionEmbeddingType


@dataclass(frozen=True)
class ChatOptions:
    """Model chat command options."""

    model_path: str
    prompt: str
    dataset_name: str | None = None
    tokenizer_path: str | None = None
    version_id: str | None = None
    architecture_path: str | None = None
    weights_path: str | None = None
    max_new_tokens: int = DEFAULT_CHAT_MAX_NEW_TOKENS
    max_token_length: int = DEFAULT_MAX_TOKEN_LENGTH
    temperature: float = DEFAULT_CHAT_TEMPERATURE
    top_k: int = DEFAULT_CHAT_TOP_K
    hidden_dim: int = DEFAULT_TRAIN_HIDDEN_DIM
    num_layers: int = DEFAULT_TRAIN_NUM_LAYERS
    attention_heads: int = DEFAULT_TRAIN_ATTENTION_HEADS
    mlp_hidden_dim: int = DEFAULT_TRAIN_MLP_HIDDEN_DIM
    mlp_layers: int = DEFAULT_TRAIN_MLP_LAYERS
    dropout: float = DEFAULT_TRAIN_DROPOUT
    position_embedding_type: PositionEmbeddingType = DEFAULT_POSITION_EMBEDDING_TYPE
    vocabulary_size: int | None = None
    stream: bool = False


@dataclass(frozen=True)
class ChatResult:
    """Model chat response payload."""

    response_text: str


@runtime_checkable
class ChatTokenizer(Protocol):
    """Structural interface for tokenizers used in chat inference.

    Both VocabularyTokenizer and HuggingFaceTokenizer satisfy this protocol.
    """

    vocabulary: dict[str, int]

    def encode(self, text: str, max_token_length: int) -> list[int]:
        """Encode text to token ids."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids back into text."""
        ...
