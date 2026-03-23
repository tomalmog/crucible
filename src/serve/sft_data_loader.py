"""SFT training data loading from JSONL and Parquet files.

This module reads prompt/response pairs from JSONL or Parquet files
and validates the schema, producing typed SftExample objects for
tokenization.
"""

from __future__ import annotations

from core.errors import CrucibleSftError
from core.sft_types import SftExample
from serve.data_file_reader import read_data_rows


def load_sft_examples(data_path: str) -> list[SftExample]:
    """Read SFT examples from a JSONL or Parquet file.

    Each row must have "prompt" and "response" string fields.
    An optional "system_prompt" field is also supported.

    Args:
        data_path: Path to JSONL or Parquet file with prompt/response pairs.

    Returns:
        Validated list of SftExample objects.

    Raises:
        CrucibleSftError: If the file is missing, empty, or contains invalid rows.
    """
    try:
        rows = read_data_rows(data_path)
    except (FileNotFoundError, ImportError, OSError) as error:
        raise CrucibleSftError(str(error)) from error
    examples: list[SftExample] = []
    for line_number, row in enumerate(rows, start=1):
        examples.append(validate_sft_example(row, line_number))
    if not examples:
        raise CrucibleSftError(
            f"SFT data file at {data_path} contains no valid examples. "
            "Add at least one prompt/response pair."
        )
    return examples


def validate_sft_example(
    row: dict[str, object],
    line_number: int,
) -> SftExample:
    """Validate one row and return a typed SftExample.

    Args:
        row: Parsed row from JSONL or Parquet.
        line_number: One-based row number for error context.

    Returns:
        Validated SftExample.

    Raises:
        CrucibleSftError: If required fields are missing or empty.
    """
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise CrucibleSftError(
            f"SFT data row {line_number}: missing or empty 'prompt' field. "
            "Each row must have a non-empty string 'prompt'."
        )
    response = row.get("response")
    if not isinstance(response, str) or not response.strip():
        raise CrucibleSftError(
            f"SFT data row {line_number}: missing or empty 'response' field. "
            "Each row must have a non-empty string 'response'."
        )
    system_prompt = row.get("system_prompt")
    if system_prompt is not None and not isinstance(system_prompt, str):
        raise CrucibleSftError(
            f"SFT data row {line_number}: 'system_prompt' must be a string."
        )
    return SftExample(
        prompt=prompt.strip(),
        response=response.strip(),
        system_prompt=system_prompt.strip() if isinstance(system_prompt, str) else None,
    )
