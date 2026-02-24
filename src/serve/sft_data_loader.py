"""SFT training data loading from JSONL files.

This module reads prompt/response pairs from JSONL files and validates
the schema, producing typed SftExample objects for tokenization.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.errors import ForgeSftError
from core.sft_types import SftExample


def load_sft_examples(data_path: str) -> list[SftExample]:
    """Read SFT examples from a JSONL file.

    Each line must be a JSON object with "prompt" and "response" string
    fields. An optional "system_prompt" field is also supported.

    Args:
        data_path: Path to JSONL file with prompt/response pairs.

    Returns:
        Validated list of SftExample objects.

    Raises:
        ForgeSftError: If the file is missing, empty, or contains invalid rows.
    """
    resolved_path = Path(data_path).expanduser().resolve()
    if not resolved_path.exists():
        raise ForgeSftError(
            f"SFT data file not found at {resolved_path}. "
            "Provide a valid --sft-data-path."
        )
    try:
        lines = resolved_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ForgeSftError(
            f"Failed to read SFT data file at {resolved_path}: {error}."
        ) from error
    examples: list[SftExample] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        row = _parse_json_line(stripped, line_number, resolved_path)
        examples.append(validate_sft_example(row, line_number))
    if not examples:
        raise ForgeSftError(
            f"SFT data file at {resolved_path} contains no valid examples. "
            "Add at least one prompt/response pair."
        )
    return examples


def validate_sft_example(
    row: dict[str, object],
    line_number: int,
) -> SftExample:
    """Validate one JSONL row and return a typed SftExample.

    Args:
        row: Parsed JSON object from one JSONL line.
        line_number: One-based line number for error context.

    Returns:
        Validated SftExample.

    Raises:
        ForgeSftError: If required fields are missing or empty.
    """
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ForgeSftError(
            f"SFT data line {line_number}: missing or empty 'prompt' field. "
            "Each row must have a non-empty string 'prompt'."
        )
    response = row.get("response")
    if not isinstance(response, str) or not response.strip():
        raise ForgeSftError(
            f"SFT data line {line_number}: missing or empty 'response' field. "
            "Each row must have a non-empty string 'response'."
        )
    system_prompt = row.get("system_prompt")
    if system_prompt is not None and not isinstance(system_prompt, str):
        raise ForgeSftError(
            f"SFT data line {line_number}: 'system_prompt' must be a string."
        )
    return SftExample(
        prompt=prompt.strip(),
        response=response.strip(),
        system_prompt=system_prompt.strip() if isinstance(system_prompt, str) else None,
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
        raise ForgeSftError(
            f"SFT data line {line_number} in {file_path}: invalid JSON ({error.msg}). "
            "Fix JSONL syntax and retry."
        ) from error
    if not isinstance(parsed, dict):
        raise ForgeSftError(
            f"SFT data line {line_number} in {file_path}: expected JSON object, "
            f"got {type(parsed).__name__}."
        )
    return parsed
