"""Tests for KTO data loader."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from core.errors import ForgeKtoError
from serve.kto_data_loader import load_kto_examples, split_kto_examples
from core.kto_types import KtoExample


def test_load_kto_examples(tmp_path: Path) -> None:
    """Load valid KTO examples from JSONL."""
    data_file = tmp_path / "kto.jsonl"
    data_file.write_text(
        json.dumps({"prompt": "p1", "response": "r1", "is_desirable": True}) + "\n"
        + json.dumps({"prompt": "p2", "response": "r2", "is_desirable": False}) + "\n"
    )
    examples = load_kto_examples(str(data_file))
    assert len(examples) == 2
    assert examples[0].is_desirable is True
    assert examples[1].is_desirable is False


def test_load_kto_missing_file() -> None:
    """Missing file raises ForgeKtoError."""
    with pytest.raises(ForgeKtoError, match="not found"):
        load_kto_examples("/nonexistent/path.jsonl")


def test_split_kto_examples() -> None:
    """Split separates desirable and undesirable examples."""
    examples = [
        KtoExample(prompt="a", response="b", is_desirable=True),
        KtoExample(prompt="c", response="d", is_desirable=False),
        KtoExample(prompt="e", response="f", is_desirable=True),
    ]
    desirable, undesirable = split_kto_examples(examples)
    assert len(desirable) == 2
    assert len(undesirable) == 1
