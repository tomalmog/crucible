"""Lance dataset persistence helpers.

This module writes snapshot records to Apache Lance when available.
It also maintains JSONL mirror files for lightweight compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.constants import LANCE_DIR_NAME, RECORDS_FILE_NAME
from core.errors import CrucibleStoreError
from core.types import DataRecord
from store.record_payload import read_data_records_jsonl, write_data_records_jsonl


def write_version_payload(version_dir: Path, records: list[DataRecord]) -> bool:
    """Persist snapshot records and attempt Lance conversion.

    Args:
        version_dir: Snapshot version directory.
        records: Records to persist.

    Returns:
        ``True`` when Lance export succeeded, else ``False``.

    Raises:
        CrucibleStoreError: If JSONL persistence fails.
    """
    _write_jsonl_records(version_dir, records)
    return _try_write_lance_dataset(version_dir, records)


def read_version_payload(version_dir: Path) -> list[DataRecord]:
    """Load snapshot records from JSONL mirror file.

    Args:
        version_dir: Snapshot version directory.

    Returns:
        Parsed records in persisted order.

    Raises:
        CrucibleStoreError: If records file is missing or invalid.
    """
    records_path = version_dir / RECORDS_FILE_NAME
    if not records_path.exists():
        raise CrucibleStoreError(
            f"Failed to load snapshot at {version_dir}: missing {RECORDS_FILE_NAME}."
        )
    try:
        return read_data_records_jsonl(records_path)
    except OSError as error:
        raise CrucibleStoreError(
            f"Failed to read snapshot payload at {records_path}: {error}. "
            "Check file permissions and retry."
        ) from error
    except ValueError as error:
        raise CrucibleStoreError(
            f"Failed to parse snapshot payload at {records_path}: {error}. "
            "Recreate the dataset snapshot."
        ) from error


def _write_jsonl_records(version_dir: Path, records: list[DataRecord]) -> None:
    """Write records to JSONL mirror file.

    Args:
        version_dir: Snapshot version directory.
        records: Snapshot records.

    Raises:
        CrucibleStoreError: If write fails.
    """
    records_path = version_dir / RECORDS_FILE_NAME
    try:
        write_data_records_jsonl(records_path, records)
    except OSError as error:
        raise CrucibleStoreError(
            f"Failed to persist snapshot payload at {records_path}: {error}. "
            "Check write permissions and available disk space."
        ) from error


def _try_write_lance_dataset(version_dir: Path, records: list[DataRecord]) -> bool:
    """Attempt to write records to Apache Lance.

    Args:
        version_dir: Snapshot version directory.
        records: Snapshot records.

    Returns:
        Whether Lance export succeeded.
    """
    try:
        import lance
        import pyarrow as pa
    except ImportError:
        return False

    table = pa.table(
        {
            "record_id": [record.record_id for record in records],
            "text": [record.text for record in records],
            "source_uri": [record.metadata.source_uri for record in records],
            "language": [record.metadata.language for record in records],
            "quality_score": [record.metadata.quality_score for record in records],
            "perplexity": [record.metadata.perplexity for record in records],
            "extra_fields": [json.dumps(record.metadata.extra_fields) for record in records],
        }
    )
    lance_uri = str(version_dir / LANCE_DIR_NAME)
    try:
        lance.write_dataset(table, lance_uri, mode="overwrite")
    except Exception as error:
        raise CrucibleStoreError(
            f"Failed to write Lance dataset at {lance_uri}: {error}. "
            "Validate lance/pyarrow compatibility and retry ingest."
        ) from error
    return True
