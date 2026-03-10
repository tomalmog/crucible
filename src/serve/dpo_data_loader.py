"""DPO training data loading from JSONL files.

This module reads preference pairs from JSONL files and validates
the schema, producing typed DpoExample objects for tokenization.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.dpo_types import DpoExample
from core.errors import CrucibleDpoError


def load_dpo_examples(data_path: str) -> list[DpoExample]:
    """Read DPO examples from a JSONL file.

    Each line must be a JSON object with "prompt", "chosen", and
    "rejected" string fields.

    Args:
        data_path: Path to JSONL file with preference pairs.

    Returns:
        Validated list of DpoExample objects.

    Raises:
        CrucibleDpoError: If the file is missing, empty, or contains invalid rows.
    """
    resolved_path = Path(data_path).expanduser().resolve()
    if not resolved_path.exists():
        raise CrucibleDpoError(
            f"DPO data file not found at {resolved_path}. "
            "Provide a valid --dpo-data-path."
        )
    try:
        lines = resolved_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CrucibleDpoError(
            f"Failed to read DPO data file at {resolved_path}: {error}."
        ) from error
    examples: list[DpoExample] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        row = _parse_json_line(stripped, line_number, resolved_path)
        examples.append(validate_dpo_example(row, line_number))
    if not examples:
        raise CrucibleDpoError(
            f"DPO data file at {resolved_path} contains no valid examples. "
            "Add at least one prompt/chosen/rejected triple."
        )
    return examples


def validate_dpo_example(
    row: dict[str, object],
    line_number: int,
) -> DpoExample:
    """Validate one JSONL row and return a typed DpoExample.

    Args:
        row: Parsed JSON object from one JSONL line.
        line_number: One-based line number for error context.

    Returns:
        Validated DpoExample.

    Raises:
        CrucibleDpoError: If required fields are missing or empty.
    """
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise CrucibleDpoError(
            f"DPO data line {line_number}: missing or empty 'prompt' field. "
            "Each row must have a non-empty string 'prompt'."
        )
    chosen = row.get("chosen")
    if not isinstance(chosen, str) or not chosen.strip():
        raise CrucibleDpoError(
            f"DPO data line {line_number}: missing or empty 'chosen' field. "
            "Each row must have a non-empty string 'chosen'."
        )
    rejected = row.get("rejected")
    if not isinstance(rejected, str) or not rejected.strip():
        raise CrucibleDpoError(
            f"DPO data line {line_number}: missing or empty 'rejected' field. "
            "Each row must have a non-empty string 'rejected'."
        )
    return DpoExample(
        prompt=prompt.strip(),
        chosen=chosen.strip(),
        rejected=rejected.strip(),
    )


def _parse_json_line(
    line: str,
    line_number: int,
    file_path: Path,
) -> dict[str, object]:
    """Parse one JSON line into a dictionary."""
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as error:
        raise CrucibleDpoError(
            f"DPO data line {line_number} in {file_path}: "
            f"invalid JSON ({error.msg}). Fix JSONL syntax and retry."
        ) from error
    if not isinstance(parsed, dict):
        raise CrucibleDpoError(
            f"DPO data line {line_number} in {file_path}: "
            f"expected JSON object, got {type(parsed).__name__}."
        )
    return parsed
