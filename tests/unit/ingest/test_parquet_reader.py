"""Unit tests for parquet_reader — covers column detection, empty-column errors,
null values, and format edge cases that could silently produce 0 records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.errors import CrucibleDependencyError, CrucibleIngestError


def _make_parquet(tmp_path: Path, data: dict[str, list]) -> Path:
    """Write a parquet file from a dict of column→values."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pytest.skip("pyarrow not installed")

    table = pa.table(data)
    path = tmp_path / "test.parquet"
    pq.write_table(table, path)
    return path


# ── Happy paths ─────────────────────────────────────────────────────────────


def test_text_column_reads_records(tmp_path: Path) -> None:
    """Standard 'text' column is extracted correctly."""
    path = _make_parquet(tmp_path, {"text": ["hello world", "foo bar", "baz qux"]})
    from ingest.parquet_reader import read_parquet_records
    records = read_parquet_records(path)
    assert len(records) == 3
    assert records[0].text == "hello world"
    assert all(r.source_uri.startswith(str(path)) for r in records)


def test_content_column_fallback(tmp_path: Path) -> None:
    """'content' column is accepted as alternative to 'text'."""
    path = _make_parquet(tmp_path, {"content": ["alpha", "beta"]})
    from ingest.parquet_reader import read_parquet_records
    records = read_parquet_records(path)
    assert len(records) == 2
    assert records[0].text == "alpha"


def test_prompt_response_pair(tmp_path: Path) -> None:
    """'prompt'+'response' pair is combined into a single record."""
    path = _make_parquet(tmp_path, {
        "prompt": ["what is 2+2?", "capital of France?"],
        "response": ["4", "Paris"],
    })
    from ingest.parquet_reader import read_parquet_records
    records = read_parquet_records(path)
    assert len(records) == 2
    assert "what is 2+2?" in records[0].text
    assert "4" in records[0].text


def test_instruction_output_pair(tmp_path: Path) -> None:
    """'instruction'+'output' is also a valid pair."""
    path = _make_parquet(tmp_path, {
        "instruction": ["solve x=3", "define recursion"],
        "output": ["x=3", "see recursion"],
    })
    from ingest.parquet_reader import read_parquet_records
    records = read_parquet_records(path)
    assert len(records) == 2


def test_source_uri_contains_row_index(tmp_path: Path) -> None:
    """source_uri encodes file path and row index for traceability."""
    path = _make_parquet(tmp_path, {"text": ["a", "b", "c"]})
    from ingest.parquet_reader import read_parquet_records
    records = read_parquet_records(path)
    uris = [r.source_uri for r in records]
    assert f"{path}:0" in uris
    assert f"{path}:1" in uris
    assert f"{path}:2" in uris


# ── Empty / null value handling ─────────────────────────────────────────────


def test_all_null_text_column_raises(tmp_path: Path) -> None:
    """All-null text column must raise CrucibleIngestError, not silently return []."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pytest.skip("pyarrow not installed")

    table = pa.table({"text": pa.array([None, None, None], type=pa.string())})
    path = tmp_path / "nulls.parquet"
    pq.write_table(table, path)

    from ingest.parquet_reader import read_parquet_records
    with pytest.raises(CrucibleIngestError, match="empty or null"):
        read_parquet_records(path)


def test_all_empty_string_text_column_raises(tmp_path: Path) -> None:
    """All-empty-string text column must raise, not return [] silently."""
    path = _make_parquet(tmp_path, {"text": ["", "   ", "\t\n"]})
    from ingest.parquet_reader import read_parquet_records
    with pytest.raises(CrucibleIngestError, match="empty or null"):
        read_parquet_records(path)


def test_mixed_null_and_valid_text_skips_nulls(tmp_path: Path) -> None:
    """Rows with null/empty text are skipped; valid rows are returned."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pytest.skip("pyarrow not installed")

    table = pa.table({"text": pa.array(["good", None, "also good", ""], type=pa.string())})
    path = tmp_path / "mixed.parquet"
    pq.write_table(table, path)

    from ingest.parquet_reader import read_parquet_records
    records = read_parquet_records(path)
    assert len(records) == 2
    texts = [r.text for r in records]
    assert "good" in texts
    assert "also good" in texts


def test_all_null_prompt_response_pair_raises(tmp_path: Path) -> None:
    """All-null prompt/response pair must raise, not silently return []."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pytest.skip("pyarrow not installed")

    table = pa.table({
        "prompt": pa.array([None, None], type=pa.string()),
        "response": pa.array([None, None], type=pa.string()),
    })
    path = tmp_path / "nullpair.parquet"
    pq.write_table(table, path)

    from ingest.parquet_reader import read_parquet_records
    with pytest.raises(CrucibleIngestError, match="empty or null"):
        read_parquet_records(path)


def test_empty_prompt_skips_row(tmp_path: Path) -> None:
    """Rows with empty prompt are skipped even if response is non-empty."""
    path = _make_parquet(tmp_path, {
        "prompt": ["valid question", ""],
        "response": ["answer", "should be skipped"],
    })
    from ingest.parquet_reader import read_parquet_records
    records = read_parquet_records(path)
    assert len(records) == 1
    assert "valid question" in records[0].text


# ── Error cases ──────────────────────────────────────────────────────────────


def test_no_recognized_column_raises(tmp_path: Path) -> None:
    """Files with no recognized columns raise CrucibleIngestError listing the columns."""
    path = _make_parquet(tmp_path, {"unknown_col": ["a", "b"], "another_col": ["c", "d"]})
    from ingest.parquet_reader import read_parquet_records
    with pytest.raises(CrucibleIngestError, match="Cannot identify a text column"):
        read_parquet_records(path)


def test_error_message_lists_columns_found(tmp_path: Path) -> None:
    """Error message names the columns present so users know what's wrong."""
    path = _make_parquet(tmp_path, {"my_col": ["x"], "other": ["y"]})
    from ingest.parquet_reader import read_parquet_records
    with pytest.raises(CrucibleIngestError, match="my_col"):
        read_parquet_records(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    """Missing .parquet file raises an error, not AttributeError."""
    from ingest.parquet_reader import read_parquet_records
    with pytest.raises(Exception):
        read_parquet_records(tmp_path / "nonexistent.parquet")


def test_only_response_column_no_prompt_raises(tmp_path: Path) -> None:
    """Having 'response' but no 'prompt' doesn't silently produce 0 records."""
    path = _make_parquet(tmp_path, {"response": ["answer1", "answer2"]})
    from ingest.parquet_reader import read_parquet_records
    with pytest.raises(CrucibleIngestError):
        read_parquet_records(path)


# ── Edge cases ───────────────────────────────────────────────────────────────


def test_sentence_column_works(tmp_path: Path) -> None:
    """'sentence' is an accepted text column name."""
    path = _make_parquet(tmp_path, {"sentence": ["The cat sat.", "Dogs bark."]})
    from ingest.parquet_reader import read_parquet_records
    records = read_parquet_records(path)
    assert len(records) == 2


def test_question_answer_pair(tmp_path: Path) -> None:
    """'question'+'answer' pair is extracted correctly."""
    path = _make_parquet(tmp_path, {
        "question": ["Q1", "Q2"],
        "answer": ["A1", "A2"],
    })
    from ingest.parquet_reader import read_parquet_records
    records = read_parquet_records(path)
    assert len(records) == 2


def test_text_column_takes_priority_over_prompt_response(tmp_path: Path) -> None:
    """'text' column is used even if 'prompt'/'response' also exist."""
    path = _make_parquet(tmp_path, {
        "text": ["direct text"],
        "prompt": ["p"],
        "response": ["r"],
    })
    from ingest.parquet_reader import read_parquet_records
    records = read_parquet_records(path)
    assert records[0].text == "direct text"


def test_large_file_returns_all_records(tmp_path: Path) -> None:
    """Reader returns all rows for a larger dataset."""
    texts = [f"row {i} content here" for i in range(500)]
    path = _make_parquet(tmp_path, {"text": texts})
    from ingest.parquet_reader import read_parquet_records
    records = read_parquet_records(path)
    assert len(records) == 500


def test_unicode_content_preserved(tmp_path: Path) -> None:
    """Unicode text (CJK, emoji, RTL) is preserved exactly."""
    path = _make_parquet(tmp_path, {
        "text": ["日本語テスト", "emoji 🔥🚀", "العربية", "Ñoño"]
    })
    from ingest.parquet_reader import read_parquet_records
    records = read_parquet_records(path)
    texts = [r.text for r in records]
    assert "日本語テスト" in texts
    assert "emoji 🔥🚀" in texts
    assert "العربية" in texts
    assert "Ñoño" in texts
