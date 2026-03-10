"""KTO data loading for unpaired preference data.

This module loads KTO-format data where each example has a binary
desirable/undesirable label rather than paired preferences.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.errors import CrucibleKtoError
from core.kto_types import KtoExample


def load_kto_examples(data_path: str) -> list[KtoExample]:
    """Load KTO examples from a JSONL file.

    Each line must have 'prompt', 'response', and 'is_desirable' fields.
    """
    path = Path(data_path)
    if not path.exists():
        raise CrucibleKtoError(f"KTO data file not found: {data_path}")
    examples: list[KtoExample] = []
    with open(path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            response = obj.get("response", "")
            is_desirable = obj.get("is_desirable", True)
            if not prompt or not response:
                continue
            examples.append(KtoExample(
                prompt=prompt,
                response=response,
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
