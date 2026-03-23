"""DPO training data loading from JSONL and Parquet files.

This module reads preference pairs from JSONL or Parquet files and
validates the schema, producing typed DpoExample objects for
tokenization.
"""

from __future__ import annotations

from core.dpo_types import DpoExample
from core.errors import CrucibleDpoError
from serve.data_file_reader import read_data_rows


def load_dpo_examples(data_path: str) -> list[DpoExample]:
    """Read DPO examples from a JSONL or Parquet file.

    Each row must have "prompt", "chosen", and "rejected" string fields.

    Args:
        data_path: Path to JSONL or Parquet file with preference pairs.

    Returns:
        Validated list of DpoExample objects.

    Raises:
        CrucibleDpoError: If the file is missing, empty, or contains invalid rows.
    """
    try:
        rows = read_data_rows(data_path)
    except (FileNotFoundError, ImportError, OSError) as error:
        raise CrucibleDpoError(str(error)) from error
    examples: list[DpoExample] = []
    for line_number, row in enumerate(rows, start=1):
        examples.append(validate_dpo_example(row, line_number))
    if not examples:
        raise CrucibleDpoError(
            f"DPO data file at {data_path} contains no valid examples. "
            "Add at least one prompt/chosen/rejected triple."
        )
    return examples


def validate_dpo_example(
    row: dict[str, object],
    line_number: int,
) -> DpoExample:
    """Validate one row and return a typed DpoExample.

    Args:
        row: Parsed row from JSONL or Parquet.
        line_number: One-based row number for error context.

    Returns:
        Validated DpoExample.

    Raises:
        CrucibleDpoError: If required fields are missing or empty.
    """
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise CrucibleDpoError(
            f"DPO data row {line_number}: missing or empty 'prompt' field. "
            "Each row must have a non-empty string 'prompt'."
        )
    chosen = row.get("chosen")
    if not isinstance(chosen, str) or not chosen.strip():
        raise CrucibleDpoError(
            f"DPO data row {line_number}: missing or empty 'chosen' field. "
            "Each row must have a non-empty string 'chosen'."
        )
    rejected = row.get("rejected")
    if not isinstance(rejected, str) or not rejected.strip():
        raise CrucibleDpoError(
            f"DPO data row {line_number}: missing or empty 'rejected' field. "
            "Each row must have a non-empty string 'rejected'."
        )
    return DpoExample(
        prompt=prompt.strip(),
        chosen=chosen.strip(),
        rejected=rejected.strip(),
    )
