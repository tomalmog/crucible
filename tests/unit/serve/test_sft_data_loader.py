"""Unit tests for SFT data loading from JSONL files."""

from __future__ import annotations

import json

import pytest

from core.errors import CrucibleSftError
from core.sft_types import SftExample
from serve.sft_data_loader import load_sft_examples, validate_sft_example


def test_load_sft_examples_valid_jsonl(tmp_path) -> None:
    """Valid JSONL file should produce correct SftExample objects."""
    data_file = tmp_path / "sft_data.jsonl"
    rows = [
        {"prompt": "What is Python?", "response": "A programming language."},
        {"prompt": "What is Rust?", "response": "A systems language."},
    ]
    data_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    examples = load_sft_examples(str(data_file))

    assert len(examples) == 2
    assert examples[0].prompt == "What is Python?"
    assert examples[0].response == "A programming language."
    assert examples[0].system_prompt is None
    assert examples[1].prompt == "What is Rust?"


def test_load_sft_examples_missing_prompt_raises(tmp_path) -> None:
    """Rows missing 'prompt' should raise CrucibleSftError with line number."""
    data_file = tmp_path / "sft_data.jsonl"
    rows = [
        {"response": "Missing prompt field."},
    ]
    data_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CrucibleSftError, match="line 1.*prompt"):
        load_sft_examples(str(data_file))


def test_load_sft_examples_empty_response_raises(tmp_path) -> None:
    """Rows with empty 'response' should raise CrucibleSftError."""
    data_file = tmp_path / "sft_data.jsonl"
    rows = [
        {"prompt": "A question", "response": "   "},
    ]
    data_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CrucibleSftError, match="line 1.*response"):
        load_sft_examples(str(data_file))


def test_validate_sft_example_with_system_prompt() -> None:
    """System prompt should be preserved when provided."""
    row: dict[str, object] = {
        "prompt": "What is AI?",
        "response": "Artificial intelligence.",
        "system_prompt": "You are a helpful assistant.",
    }
    example = validate_sft_example(row, line_number=1)

    assert example == SftExample(
        prompt="What is AI?",
        response="Artificial intelligence.",
        system_prompt="You are a helpful assistant.",
    )
