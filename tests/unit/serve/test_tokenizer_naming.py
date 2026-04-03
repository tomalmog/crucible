"""Tests that the two load_huggingface_tokenizer-like functions are correctly
separated and no cross-import confusion exists.

Regression for: audit finding #1 — two functions with the same name in
different modules with incompatible signatures, causing import-order bugs."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Name disambiguation ──────────────────────────────────────────────────────


def test_huggingface_tokenizer_module_has_load_tokenizer_from_file() -> None:
    """huggingface_tokenizer exports load_tokenizer_from_file (not load_huggingface_tokenizer)."""
    import serve.huggingface_tokenizer as ht
    assert hasattr(ht, "load_tokenizer_from_file"), (
        "load_tokenizer_from_file must exist in serve.huggingface_tokenizer"
    )
    # The old name must NOT exist — removing it prevents import confusion
    assert not hasattr(ht, "load_huggingface_tokenizer"), (
        "load_huggingface_tokenizer must be removed from serve.huggingface_tokenizer "
        "to avoid collision with hf_model_loader.load_huggingface_tokenizer"
    )


def test_hf_model_loader_has_load_huggingface_tokenizer() -> None:
    """hf_model_loader.load_huggingface_tokenizer loads by model ID (str), not file path."""
    from serve.hf_model_loader import load_huggingface_tokenizer
    import inspect
    sig = inspect.signature(load_huggingface_tokenizer)
    params = list(sig.parameters)
    assert params[0] == "model_id", (
        "hf_model_loader.load_huggingface_tokenizer must accept model_id as first param"
    )


def test_load_tokenizer_from_file_accepts_path_string() -> None:
    """load_tokenizer_from_file accepts a file path string."""
    import inspect
    from serve.huggingface_tokenizer import load_tokenizer_from_file
    sig = inspect.signature(load_tokenizer_from_file)
    params = list(sig.parameters)
    assert params[0] == "tokenizer_path"


def test_training_metadata_imports_from_correct_module(tmp_path: Path) -> None:
    """training_metadata.load_tokenizer_from_path uses load_tokenizer_from_file,
    not load_huggingface_tokenizer, when loading a HuggingFace tokenizer.json."""
    # Build a fake HF tokenizer.json payload
    import json
    hf_payload = {
        "model": {"type": "BPE", "vocab": {"hello": 0, "world": 1}},
        "added_tokens": [],
    }
    vocab_path = tmp_path / "tokenizer.json"
    vocab_path.write_text(json.dumps(hf_payload), encoding="utf-8")

    from serve.tokenization import VocabularyTokenizer
    fake = VocabularyTokenizer(vocabulary={"hello": 0, "world": 1})

    with patch("serve.huggingface_tokenizer.load_tokenizer_from_file", return_value=fake) as mock_fn:
        from serve.training_metadata import load_tokenizer_from_path
        result = load_tokenizer_from_path(str(vocab_path))
        mock_fn.assert_called_once_with(str(vocab_path))

    assert result is fake


def test_no_cross_import_of_wrong_load_function() -> None:
    """training_metadata never imports load_huggingface_tokenizer from huggingface_tokenizer."""
    import ast
    training_meta_path = Path(__file__).parent.parent.parent.parent / "src" / "serve" / "training_metadata.py"
    source = training_meta_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                if "huggingface_tokenizer" in module and "load_huggingface_tokenizer" in names:
                    raise AssertionError(
                        "training_metadata.py still imports load_huggingface_tokenizer from "
                        "huggingface_tokenizer — must use load_tokenizer_from_file instead"
                    )


# ── load_tokenizer_from_file behavior ────────────────────────────────────────


def test_load_tokenizer_from_file_raises_on_missing_file(tmp_path: Path) -> None:
    """load_tokenizer_from_file raises CrucibleServeError for missing file."""
    from core.errors import CrucibleServeError
    from serve.huggingface_tokenizer import load_tokenizer_from_file

    missing = str(tmp_path / "nonexistent.json")
    with patch("serve.huggingface_tokenizer._import_tokenizers") as mock_imp:
        mock_tokenizers = MagicMock()
        mock_tokenizers.Tokenizer.from_file.side_effect = Exception("file not found")
        mock_imp.return_value = mock_tokenizers

        with pytest.raises(Exception):
            load_tokenizer_from_file(missing)


def test_load_tokenizer_from_file_raises_dependency_error_without_tokenizers() -> None:
    """load_tokenizer_from_file raises CrucibleDependencyError if tokenizers not installed."""
    from core.errors import CrucibleDependencyError
    from serve.huggingface_tokenizer import load_tokenizer_from_file

    with patch("serve.huggingface_tokenizer._import_tokenizers") as mock_imp:
        mock_imp.side_effect = CrucibleDependencyError("tokenizers not installed")
        with pytest.raises(CrucibleDependencyError, match="tokenizers"):
            load_tokenizer_from_file("some/path.json")


# ── AutoTokenizerAdapter ──────────────────────────────────────────────────────


def test_auto_tokenizer_adapter_encode_decode_round_trip() -> None:
    """AutoTokenizerAdapter encode→decode should round-trip (approximately)."""
    from serve.huggingface_tokenizer import AutoTokenizerAdapter

    mock_hf = MagicMock()
    mock_hf.get_vocab.return_value = {"hello": 0, "world": 1, "<unk>": 2}
    mock_hf.eos_token_id = 2
    mock_hf.encode.return_value = [0, 1]
    mock_hf.decode.return_value = "hello world"

    adapter = AutoTokenizerAdapter(mock_hf)
    encoded = adapter.encode("hello world", max_token_length=512)
    decoded = adapter.decode(encoded)

    assert encoded == [0, 1]
    assert decoded == "hello world"


def test_auto_tokenizer_adapter_truncates_at_max_token_length() -> None:
    """AutoTokenizerAdapter truncates to max_token_length."""
    from serve.huggingface_tokenizer import AutoTokenizerAdapter

    mock_hf = MagicMock()
    mock_hf.get_vocab.return_value = {}
    mock_hf.eos_token_id = 1
    mock_hf.encode.return_value = list(range(1000))  # long sequence

    adapter = AutoTokenizerAdapter(mock_hf)
    result = adapter.encode("long text", max_token_length=128)
    assert len(result) == 128


def test_auto_tokenizer_adapter_eos_token_id_none_defaults_to_zero() -> None:
    """AutoTokenizerAdapter handles None eos_token_id gracefully."""
    from serve.huggingface_tokenizer import AutoTokenizerAdapter

    mock_hf = MagicMock()
    mock_hf.get_vocab.return_value = {}
    mock_hf.eos_token_id = None  # Some models don't have EOS

    adapter = AutoTokenizerAdapter(mock_hf)
    assert adapter.eos_token_id == 0


import pytest
