"""Shared JSONL serialization for DataRecord payloads.

This module centralizes DataRecord JSON serialization logic.
It is reused by snapshot persistence and ingest checkpoint flows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.types import DataRecord, RecordMetadata


def data_record_to_payload(record: DataRecord) -> dict[str, object]:
    """Serialize DataRecord into JSON-safe payload.

    Args:
        record: Data record instance.

    Returns:
        Dictionary payload for JSON encoding.
    """
    metadata = record.metadata
    return {
        "record_id": record.record_id,
        "text": record.text,
        "metadata": {
            "source_uri": metadata.source_uri,
            "language": metadata.language,
            "quality_score": metadata.quality_score,
            "perplexity": metadata.perplexity,
            "extra_fields": dict(metadata.extra_fields),
        },
    }


def data_record_from_payload(payload: dict[str, Any]) -> DataRecord:
    """Deserialize JSON payload into DataRecord.

    Args:
        payload: Serialized record payload.

    Returns:
        Parsed DataRecord.
    """
    metadata_payload = payload["metadata"]
    metadata_dict = metadata_payload if isinstance(metadata_payload, dict) else {}
    metadata = RecordMetadata(
        source_uri=str(metadata_dict.get("source_uri", "")),
        language=str(metadata_dict.get("language", "unknown")),
        quality_score=float(metadata_dict.get("quality_score", 0.0)),
        perplexity=float(metadata_dict.get("perplexity", 0.0)),
        extra_fields={
            str(key): str(value)
            for key, value in dict(metadata_dict.get("extra_fields", {})).items()
        },
    )
    return DataRecord(
        record_id=str(payload.get("record_id", "")),
        text=str(payload.get("text", "")),
        metadata=metadata,
    )


def write_data_records_jsonl(records_path: Path, records: list[DataRecord]) -> None:
    """Write DataRecord list to JSONL file.

    Args:
        records_path: Output JSONL file path.
        records: Records to serialize.
    """
    lines = [json.dumps(data_record_to_payload(record), sort_keys=True) for record in records]
    records_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_data_records_jsonl(records_path: Path) -> list[DataRecord]:
    """Read DataRecord list from JSONL file.

    Args:
        records_path: Input JSONL file path.

    Returns:
        Parsed records.

    Raises:
        ValueError: If JSONL rows are invalid.
    """
    parsed_records: list[DataRecord] = []
    with open(records_path, encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            payload = _parse_payload_line(line, line_number)
            parsed_records.append(data_record_from_payload(payload))
    return parsed_records


def _parse_payload_line(line: str, line_number: int) -> dict[str, Any]:
    """Parse and validate one JSONL payload row.

    Args:
        line: Raw JSONL line.
        line_number: One-based line number.

    Returns:
        Parsed payload dictionary.

    Raises:
        ValueError: If JSON row is invalid.
    """
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON at line {line_number}: {error.msg}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid payload at line {line_number}: expected JSON object")
    return payload
