"""Persistence helpers for training configuration and tokenizer artifacts.

This module stores and reloads metadata that keeps training and chat
runtime settings consistent across separate commands.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

from core.chat_types import ChatTokenizer
from core.constants import (
    DEFAULT_TOKENIZER_VOCAB_FILE_NAME,
    DEFAULT_TRAINING_CONFIG_FILE_NAME,
)
from core.errors import ForgeServeError
from core.types import TrainingOptions
from serve.tokenization import VocabularyTokenizer


def save_training_config(output_dir: Path, options: TrainingOptions) -> Path:
    """Persist training options used to build the model architecture."""
    config_path = output_dir / DEFAULT_TRAINING_CONFIG_FILE_NAME
    config_path.write_text(json.dumps(asdict(options), indent=2) + "\n", encoding="utf-8")
    return config_path


def save_tokenizer_vocabulary(output_dir: Path, tokenizer: VocabularyTokenizer) -> Path:
    """Persist fitted tokenizer vocabulary used during training."""
    vocabulary_path = output_dir / DEFAULT_TOKENIZER_VOCAB_FILE_NAME
    payload = {
        token: token_id
        for token, token_id in sorted(tokenizer.vocabulary.items(), key=_sort_vocabulary_item)
    }
    vocabulary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return vocabulary_path


def load_training_config(model_path: str) -> dict[str, object] | None:
    """Load persisted training config located beside model weights."""
    config_path = _artifact_dir(model_path) / DEFAULT_TRAINING_CONFIG_FILE_NAME
    if not config_path.exists():
        return None
    payload = _read_json_payload(config_path)
    if not isinstance(payload, dict):
        raise ForgeServeError(
            f"Invalid training config format at {config_path}: expected JSON object."
        )
    return cast(dict[str, object], payload)


def load_tokenizer(model_path: str) -> ChatTokenizer | None:
    """Load persisted training tokenizer located beside model weights.

    Checks for Forge vocab.json first, then HuggingFace tokenizer.json.
    Returns None if no vocabulary file exists next to the model.
    """
    artifact_dir = _artifact_dir(model_path)
    vocabulary_path = artifact_dir / DEFAULT_TOKENIZER_VOCAB_FILE_NAME
    if vocabulary_path.exists():
        return load_tokenizer_from_path(str(vocabulary_path))
    hf_tokenizer_path = artifact_dir / "tokenizer.json"
    if hf_tokenizer_path.exists():
        return load_tokenizer_from_path(str(hf_tokenizer_path))
    return None


def load_tokenizer_from_path(vocabulary_path: str) -> ChatTokenizer:
    """Load tokenizer from an explicit file path.

    Supports two formats:
    - Forge flat vocabulary: ``{"token": id, ...}``
    - HuggingFace tokenizer.json: loaded via the ``tokenizers`` library for
      full BPE/WordPiece/Unigram encode/decode support.

    Args:
        vocabulary_path: Absolute or relative path to a JSON vocabulary file.

    Returns:
        Loaded tokenizer instance.

    Raises:
        ForgeServeError: If the file is missing, malformed, or has invalid entries.
    """
    resolved_path = Path(vocabulary_path).expanduser().resolve()
    if not resolved_path.exists():
        raise ForgeServeError(
            f"Tokenizer vocabulary file not found at {resolved_path}. "
            "Provide a valid --tokenizer-path or re-run training to generate vocab.json."
        )
    payload = _read_json_payload(resolved_path)
    if not isinstance(payload, dict):
        raise ForgeServeError(
            f"Invalid tokenizer vocabulary format at {resolved_path}: expected JSON object."
        )
    if _is_huggingface_tokenizer(payload):
        from serve.huggingface_tokenizer import load_huggingface_tokenizer

        return load_huggingface_tokenizer(str(resolved_path))
    raw_vocab = _extract_vocabulary_mapping(payload, resolved_path)
    return _validate_vocabulary(raw_vocab, resolved_path)


def _is_huggingface_tokenizer(payload: dict[str, object]) -> bool:
    """Return True if the payload looks like a HuggingFace tokenizer.json."""
    model_section = payload.get("model")
    return isinstance(model_section, dict) and "vocab" in model_section


def _extract_vocabulary_mapping(
    payload: dict[str, object],
    source_path: Path,
) -> dict[str, object]:
    """Extract the token-to-id mapping from a flat Forge vocabulary payload."""
    if _looks_like_flat_vocabulary(payload):
        return payload
    raise ForgeServeError(
        f"Unrecognized tokenizer format at {source_path}. "
        "Expected a Forge vocab.json (flat {{token: id}}) or "
        "a HuggingFace tokenizer.json. Install the tokenizers library "
        "for HuggingFace support: pip install tokenizers"
    )


def _looks_like_flat_vocabulary(payload: dict[str, object]) -> bool:
    """Return True if the payload looks like a flat token-to-id mapping."""
    for value in list(payload.values())[:5]:
        if not isinstance(value, int):
            return False
    return True


def _validate_vocabulary(
    raw_vocab: dict[str, object],
    source_path: Path,
) -> VocabularyTokenizer:
    """Validate and build a VocabularyTokenizer from a raw mapping."""
    vocabulary: dict[str, int] = {}
    for raw_token, raw_token_id in raw_vocab.items():
        if not isinstance(raw_token, str):
            raise ForgeServeError(
                f"Invalid tokenizer vocabulary at {source_path}: token keys must be strings."
            )
        if not isinstance(raw_token_id, int):
            raise ForgeServeError(
                f"Invalid tokenizer vocabulary at {source_path}: token id for "
                f"{raw_token!r} must be an integer."
            )
        vocabulary[raw_token] = raw_token_id
    return VocabularyTokenizer(vocabulary=vocabulary)


def _artifact_dir(model_path: str) -> Path:
    """Resolve output directory containing model and metadata artifacts."""
    resolved = Path(model_path).expanduser().resolve()
    if resolved.is_dir():
        return resolved
    return resolved.parent


def _read_json_payload(payload_path: Path) -> object:
    """Read JSON payload from disk with traceable errors."""
    try:
        return json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ForgeServeError(
            f"Failed to parse JSON artifact at {payload_path}: {error.msg}. "
            "Re-run training to regenerate valid artifacts."
        ) from error
    except OSError as error:
        raise ForgeServeError(
            f"Failed to read artifact at {payload_path}: {error}. "
            "Verify file permissions and retry."
        ) from error


def _sort_vocabulary_item(item: tuple[str, int]) -> tuple[int, str]:
    """Sort vocabulary output by token id for deterministic storage."""
    token, token_id = item
    return token_id, token
