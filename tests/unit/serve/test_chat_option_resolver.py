"""Unit tests for chat option resolution helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.chat_types import ChatOptions
from core.errors import CrucibleServeError
from core.types import DataRecord, RecordMetadata, TrainingOptions
from serve.chat_option_resolver import resolve_chat_tokenizer, resolve_chat_training_options
from serve.tokenization import VocabularyTokenizer
from serve.training_metadata import save_tokenizer_vocabulary, save_training_config


def test_resolve_chat_training_options_prefers_persisted_config(tmp_path: Path) -> None:
    """Persisted training config should drive chat model options."""
    save_training_config(
        tmp_path,
        TrainingOptions(
            dataset_name="demo",
            output_dir=str(tmp_path),
            hidden_dim=320,
            position_embedding_type="sinusoidal",
            max_token_length=300,
        ),
    )
    model_path = tmp_path / "model.pt"
    model_path.write_text("placeholder", encoding="utf-8")
    options = ChatOptions(model_path=str(model_path), prompt="hello", dataset_name="demo")

    resolved = resolve_chat_training_options(options, model_state={})

    assert (
        resolved.hidden_dim == 320
        and resolved.position_embedding_type == "sinusoidal"
        and resolved.max_token_length == 300
    )


def test_resolve_chat_tokenizer_prefers_persisted_vocabulary(tmp_path: Path) -> None:
    """Persisted tokenizer should be reused instead of refitting records."""
    save_tokenizer_vocabulary(
        tmp_path,
        VocabularyTokenizer(vocabulary={"<pad>": 0, "<unk>": 1, "persisted": 2}),
    )
    model_path = tmp_path / "model.pt"
    model_path.write_text("placeholder", encoding="utf-8")
    options = ChatOptions(model_path=str(model_path), prompt="hello", dataset_name="demo")
    records = [
        DataRecord(
            record_id="r-1",
            text="fresh tokens only",
            metadata=RecordMetadata(
                source_uri="data.txt",
                language="en",
                quality_score=0.5,
                perplexity=1.0,
            ),
        )
    ]
    training_options = TrainingOptions(dataset_name="demo", output_dir=str(tmp_path))

    tokenizer = resolve_chat_tokenizer(records, options, training_options)

    assert tokenizer.vocabulary == {"<pad>": 0, "<unk>": 1, "persisted": 2}


def test_resolve_chat_tokenizer_uses_explicit_tokenizer_path(tmp_path: Path) -> None:
    """Explicit tokenizer path should take priority over all other sources."""
    vocab_path = tmp_path / "custom_vocab.json"
    vocab_path.write_text(
        json.dumps({"<pad>": 0, "<unk>": 1, "explicit": 2}), encoding="utf-8"
    )
    model_path = tmp_path / "model.pt"
    model_path.write_text("placeholder", encoding="utf-8")
    options = ChatOptions(
        model_path=str(model_path),
        prompt="hello",
        tokenizer_path=str(vocab_path),
    )
    training_options = TrainingOptions(dataset_name="", output_dir=str(tmp_path))

    tokenizer = resolve_chat_tokenizer(None, options, training_options)

    assert tokenizer.vocabulary == {"<pad>": 0, "<unk>": 1, "explicit": 2}


def test_resolve_chat_tokenizer_with_persisted_vocab_and_no_records(tmp_path: Path) -> None:
    """Persisted vocab beside model should work when records are None."""
    save_tokenizer_vocabulary(
        tmp_path,
        VocabularyTokenizer(vocabulary={"<pad>": 0, "<unk>": 1, "saved": 2}),
    )
    model_path = tmp_path / "model.pt"
    model_path.write_text("placeholder", encoding="utf-8")
    options = ChatOptions(model_path=str(model_path), prompt="hello")
    training_options = TrainingOptions(dataset_name="", output_dir=str(tmp_path))

    tokenizer = resolve_chat_tokenizer(None, options, training_options)

    assert tokenizer.vocabulary == {"<pad>": 0, "<unk>": 1, "saved": 2}


def test_resolve_chat_tokenizer_raises_when_no_source_available(tmp_path: Path) -> None:
    """Should raise a clear error when no tokenizer source is available."""
    model_path = tmp_path / "model.pt"
    model_path.write_text("placeholder", encoding="utf-8")
    options = ChatOptions(model_path=str(model_path), prompt="hello")
    training_options = TrainingOptions(dataset_name="", output_dir=str(tmp_path))

    with pytest.raises(CrucibleServeError, match="No tokenizer found"):
        resolve_chat_tokenizer(None, options, training_options)
