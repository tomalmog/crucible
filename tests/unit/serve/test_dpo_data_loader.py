"""Unit tests for DPO data loading from JSONL files."""

from __future__ import annotations

import json

import pytest

from core.dpo_types import DpoExample
from core.errors import CrucibleDpoError
from serve.dpo_data_loader import load_dpo_examples, validate_dpo_example


def test_load_dpo_examples_valid_jsonl(tmp_path) -> None:
    """Valid JSONL file should produce correct DpoExample objects."""
    data_file = tmp_path / "dpo_data.jsonl"
    rows = [
        {"prompt": "What is Python?", "chosen": "A great language.", "rejected": "A snake."},
        {"prompt": "What is Rust?", "chosen": "A systems language.", "rejected": "Iron oxide."},
    ]
    data_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    examples = load_dpo_examples(str(data_file))

    assert len(examples) == 2
    assert examples[0].prompt == "What is Python?"
    assert examples[0].chosen == "A great language."
    assert examples[0].rejected == "A snake."
    assert examples[1].prompt == "What is Rust?"


def test_load_dpo_examples_missing_prompt_raises(tmp_path) -> None:
    """Rows missing 'prompt' should raise CrucibleDpoError with line number."""
    data_file = tmp_path / "dpo_data.jsonl"
    rows = [
        {"chosen": "Good answer.", "rejected": "Bad answer."},
    ]
    data_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CrucibleDpoError, match="line 1.*prompt"):
        load_dpo_examples(str(data_file))


def test_load_dpo_examples_missing_chosen_raises(tmp_path) -> None:
    """Rows missing 'chosen' should raise CrucibleDpoError."""
    data_file = tmp_path / "dpo_data.jsonl"
    rows = [
        {"prompt": "A question", "rejected": "Bad answer."},
    ]
    data_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CrucibleDpoError, match="line 1.*chosen"):
        load_dpo_examples(str(data_file))


def test_load_dpo_examples_empty_rejected_raises(tmp_path) -> None:
    """Rows with empty 'rejected' should raise CrucibleDpoError."""
    data_file = tmp_path / "dpo_data.jsonl"
    rows = [
        {"prompt": "A question", "chosen": "Good answer.", "rejected": "   "},
    ]
    data_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CrucibleDpoError, match="line 1.*rejected"):
        load_dpo_examples(str(data_file))


def test_load_dpo_examples_empty_file_raises(tmp_path) -> None:
    """Empty file should raise CrucibleDpoError."""
    data_file = tmp_path / "dpo_data.jsonl"
    data_file.write_text("", encoding="utf-8")

    with pytest.raises(CrucibleDpoError, match="no valid examples"):
        load_dpo_examples(str(data_file))


def test_load_dpo_examples_missing_file_raises(tmp_path) -> None:
    """Non-existent file should raise CrucibleDpoError."""
    with pytest.raises(CrucibleDpoError, match="not found"):
        load_dpo_examples(str(tmp_path / "missing.jsonl"))


def test_validate_dpo_example_strips_whitespace() -> None:
    """Fields should be stripped of leading/trailing whitespace."""
    row: dict[str, object] = {
        "prompt": "  What is AI?  ",
        "chosen": "  Artificial intelligence.  ",
        "rejected": "  I don't know.  ",
    }
    example = validate_dpo_example(row, line_number=1)

    assert example == DpoExample(
        prompt="What is AI?",
        chosen="Artificial intelligence.",
        rejected="I don't know.",
    )
