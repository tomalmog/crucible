"""Parquet file reader for dataset ingestion.

Extracts text records from ``.parquet`` files by probing common
column names used by HuggingFace datasets and instruction-tuning
corpora.
"""

from __future__ import annotations

from pathlib import Path

from core.errors import CrucibleDependencyError, CrucibleIngestError
from core.types import SourceTextRecord

# Column names tried (in order) when extracting text from parquet rows.
_TEXT_COLUMNS = ("text", "content", "sentence", "document")
_PROMPT_COLUMNS = ("prompt", "instruction", "input", "question")
_RESPONSE_COLUMNS = ("response", "output", "answer", "completion")


def read_parquet_records(file_path: Path) -> list[SourceTextRecord]:
    """Read text records from a parquet file.

    Tries common column names for text extraction: ``text``,
    ``content``, ``sentence``, ``document``.  Falls back to
    ``prompt``/``response`` pairs if no single text column exists.

    Args:
        file_path: Path to a ``.parquet`` file.

    Returns:
        Source records extracted from parquet rows.

    Raises:
        CrucibleDependencyError: If pyarrow is not installed.
        CrucibleIngestError: If no text column can be identified.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise CrucibleDependencyError(
            "Parquet support requires pyarrow, but it is not installed. "
            "Install pyarrow to ingest .parquet files."
        ) from error

    table = pq.read_table(file_path)
    columns = {c.lower(): c for c in table.column_names}

    text_col = _find_column(columns, _TEXT_COLUMNS)
    if text_col:
        return _extract_single_column(file_path, table, text_col)

    prompt_col = _find_column(columns, _PROMPT_COLUMNS)
    response_col = _find_column(columns, _RESPONSE_COLUMNS)
    if prompt_col and response_col:
        return _extract_prompt_response(
            file_path, table, prompt_col, response_col,
        )

    raise CrucibleIngestError(
        f"Cannot identify a text column in {file_path}. "
        f"Columns found: {table.column_names}. "
        "Expected one of: text, content, sentence, document, "
        "or prompt+response pair."
    )


def _find_column(
    columns: dict[str, str], candidates: tuple[str, ...],
) -> str | None:
    """Return the original column name for the first matching candidate."""
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _extract_single_column(
    file_path: Path, table: object, column: str,
) -> list[SourceTextRecord]:
    """Extract records from a single text column."""
    import pyarrow as pa

    assert isinstance(table, pa.Table)
    series = table.column(column)
    records: list[SourceTextRecord] = []
    for idx, value in enumerate(series):
        text = value.as_py()
        if isinstance(text, str) and text.strip():
            records.append(SourceTextRecord(
                source_uri=f"{file_path}:{idx}", text=text,
            ))
    return records


def _extract_prompt_response(
    file_path: Path, table: object,
    prompt_col: str, response_col: str,
) -> list[SourceTextRecord]:
    """Extract records by combining prompt and response columns."""
    import pyarrow as pa

    assert isinstance(table, pa.Table)
    prompts = table.column(prompt_col)
    responses = table.column(response_col)
    records: list[SourceTextRecord] = []
    for idx, (prompt, response) in enumerate(zip(prompts, responses)):
        p_text = prompt.as_py()
        r_text = response.as_py()
        if isinstance(p_text, str) and isinstance(r_text, str):
            records.append(SourceTextRecord(
                source_uri=f"{file_path}:{idx}",
                text=f"{p_text}\n{r_text}",
            ))
    return records
