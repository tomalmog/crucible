"""KTO data loading for unpaired preference data.

This module loads KTO-format data where each example has a binary
desirable/undesirable label rather than paired preferences.
"""

from __future__ import annotations

from core.errors import CrucibleKtoError
from core.kto_types import KtoExample
from serve.data_file_reader import read_data_rows


def load_kto_examples(data_path: str) -> list[KtoExample]:
    """Load KTO examples from a JSONL or Parquet file.

    Each row must have 'prompt', 'response', and 'is_desirable' fields.
    """
    try:
        rows = read_data_rows(data_path)
    except (FileNotFoundError, ImportError, OSError) as error:
        raise CrucibleKtoError(str(error)) from error
    examples: list[KtoExample] = []
    for row in rows:
        prompt = row.get("prompt", "")
        response = row.get("response", "")
        is_desirable = row.get("is_desirable", True)
        if not prompt or not response:
            continue
        examples.append(KtoExample(
            prompt=str(prompt),
            response=str(response),
            is_desirable=bool(is_desirable),
        ))
    return examples


def split_kto_examples(
    examples: list[KtoExample],
) -> tuple[list[KtoExample], list[KtoExample]]:
    """Split examples into desirable and undesirable groups."""
    desirable = [e for e in examples if e.is_desirable]
    undesirable = [e for e in examples if not e.is_desirable]
    return desirable, undesirable
