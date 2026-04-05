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

_BATCH_SIZE = 10_000


def read_parquet_records(file_path: Path, text_field: str = "") -> list[SourceTextRecord]:
    """Read text records from a parquet file.

    Uses batched reading via ``ParquetFile.iter_batches()`` to avoid
    loading the entire file into memory at once.

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

    pf = pq.ParquetFile(file_path)
    col_names = pf.schema_arrow.names
    num_rows = pf.metadata.num_rows
    columns = {c.lower(): c for c in col_names}

    # Explicit text_field override
    if text_field:
        original = columns.get(text_field.lower())
        if not original:
            raise CrucibleIngestError(
                f"Column '{text_field}' not found in {file_path}. "
                f"Available columns: {col_names}."
            )
        extra_cols = [c for c in col_names if c != original]
        records = _extract_single_column_with_extras(
            file_path, pf, original, extra_cols,
        )
        if not records:
            raise CrucibleIngestError(
                f"Column '{text_field}' in {file_path} has no non-empty values."
            )
        return records

    text_col = _find_column(columns, _TEXT_COLUMNS)
    if text_col:
        records = _extract_single_column(file_path, pf, text_col)
        if not records:
            raise CrucibleIngestError(
                f"Column '{text_col}' in {file_path} exists but all values are empty or null. "
                f"Total rows: {num_rows}. Verify the file has non-empty text."
            )
        return records

    prompt_col = _find_column(columns, _PROMPT_COLUMNS)
    response_col = _find_column(columns, _RESPONSE_COLUMNS)
    if prompt_col and response_col:
        records = _extract_prompt_response(file_path, pf, prompt_col, response_col)
        if not records:
            raise CrucibleIngestError(
                f"Columns '{prompt_col}'/'{response_col}' in {file_path} exist but all rows "
                f"have empty or null values. Total rows: {num_rows}."
            )
        return records

    raise CrucibleIngestError(
        f"Cannot identify a text column in {file_path}. "
        f"Columns found: {col_names}. "
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
    file_path: Path, pf: object, column: str,
) -> list[SourceTextRecord]:
    """Extract records from a single text column using batched reading."""
    import pyarrow.parquet as pq_mod

    assert isinstance(pf, pq_mod.ParquetFile)
    records: list[SourceTextRecord] = []
    row_offset = 0
    for batch in pf.iter_batches(batch_size=_BATCH_SIZE, columns=[column]):
        series = batch.column(column)
        for idx, value in enumerate(series):
            text = value.as_py()
            if isinstance(text, str) and text.strip():
                records.append(SourceTextRecord(
                    source_uri=f"{file_path}:{row_offset + idx}", text=text,
                ))
        row_offset += batch.num_rows
    return records


def _extract_single_column_with_extras(
    file_path: Path, pf: object, text_col: str, extra_cols: list[str],
) -> list[SourceTextRecord]:
    """Extract records from a text column, preserving other columns as extras."""
    import pyarrow.parquet as pq_mod

    assert isinstance(pf, pq_mod.ParquetFile)
    all_cols = [text_col] + extra_cols
    records: list[SourceTextRecord] = []
    row_offset = 0
    for batch in pf.iter_batches(batch_size=_BATCH_SIZE, columns=all_cols):
        text_series = batch.column(text_col)
        for idx, value in enumerate(text_series):
            text = value.as_py()
            if isinstance(text, str) and text.strip():
                extras = {}
                for c in extra_cols:
                    val = batch.column(c)[idx].as_py()
                    if isinstance(val, (str, int, float, bool)):
                        extras[c] = str(val)
                records.append(SourceTextRecord(
                    source_uri=f"{file_path}:{row_offset + idx}",
                    text=text,
                    extra_fields=extras,
                ))
        row_offset += batch.num_rows
    return records


def _extract_prompt_response(
    file_path: Path, pf: object,
    prompt_col: str, response_col: str,
) -> list[SourceTextRecord]:
    """Extract records by combining prompt and response columns."""
    import pyarrow.parquet as pq_mod

    assert isinstance(pf, pq_mod.ParquetFile)
    records: list[SourceTextRecord] = []
    row_offset = 0
    for batch in pf.iter_batches(
        batch_size=_BATCH_SIZE, columns=[prompt_col, response_col],
    ):
        prompts = batch.column(prompt_col)
        responses = batch.column(response_col)
        for idx, (prompt, response) in enumerate(zip(prompts, responses)):
            p_text = prompt.as_py()
            r_text = response.as_py()
            if isinstance(p_text, str) and isinstance(r_text, str) and p_text.strip():
                records.append(SourceTextRecord(
                    source_uri=f"{file_path}:{row_offset + idx}",
                    text=f"{p_text}\n{r_text}",
                ))
        row_offset += batch.num_rows
    return records
