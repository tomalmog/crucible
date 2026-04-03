"""Shared text and label extraction for interpretability tools."""

from __future__ import annotations

from typing import Any

TEXT_ATTR_NAMES = ("text", "content", "prompt", "instruction", "input")


def extract_texts(records: list[Any], max_samples: int) -> list[str]:
    """Extract text content from dataset records of any format."""
    texts: list[str] = []
    for record in records[:max_samples]:
        text = extract_single_text(record)
        if text:
            texts.append(text)
    return texts


def extract_single_text(record: Any) -> str:
    """Extract a single text string from a record of any format."""
    for attr in TEXT_ATTR_NAMES:
        val = getattr(record, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()

    if isinstance(record, dict):
        for key in TEXT_ATTR_NAMES:
            val = record.get(key, "")
            if isinstance(val, str) and val.strip():
                return val.strip()
        for key in ("text", "content"):
            val = record.get(key)
            if isinstance(val, dict):
                inner = val.get("text", "") or val.get("input", "")
                if inner:
                    return str(inner).strip()

    return ""


def extract_column_texts(
    records: list[Any], column: str, max_samples: int,
) -> list[str]:
    """Extract text from a specific column/field of dataset records."""
    texts: list[str] = []
    for record in records[:max_samples]:
        val = _get_field_value(record, column)
        if isinstance(val, str) and val.strip():
            texts.append(val.strip())
    return texts


def _get_field_value(record: Any, field: str) -> str:
    """Get a field value from a record, checking top-level and extra_fields."""
    # Top-level attribute (e.g. record.text)
    val = getattr(record, field, None)
    if isinstance(val, str) and val.strip():
        return val

    # Dict top-level
    if isinstance(record, dict):
        val = record.get(field)
        if isinstance(val, str) and val.strip():
            return val

    # metadata.extra_fields
    meta = getattr(record, "metadata", None)
    if meta is not None:
        if isinstance(meta, dict):
            extra = meta.get("extra_fields")
            if isinstance(extra, dict):
                val = extra.get(field)
                if isinstance(val, str):
                    return val
        else:
            extra = getattr(meta, "extra_fields", None)
            if isinstance(extra, dict):
                val = extra.get(field)
                if isinstance(val, str):
                    return val
    return ""


def get_label(records: list[Any], index: int, field: str) -> str:
    """Extract a metadata label from a record."""
    if not field or index >= len(records):
        return ""
    record = records[index]
    meta = getattr(record, "metadata", None)
    if meta is None:
        return ""
    # Plain dict (e.g. _SimpleRecord from agent_entry_script)
    if isinstance(meta, dict):
        # Direct field (flat JSONL like {"text": "...", "sentiment": "pos"})
        val = meta.get(field)
        if val is not None and val != "":
            return str(val)
        # Nested records.jsonl format: metadata dict has a "metadata" key
        # with "extra_fields" inside (Crucible ingested dataset pushed remote)
        inner_meta = meta.get("metadata")
        if isinstance(inner_meta, dict):
            inner_extra = inner_meta.get("extra_fields")
            if isinstance(inner_extra, dict):
                inner_val = inner_extra.get(field)
                if inner_val is not None and inner_val != "":
                    return str(inner_val)
        return ""
    # RecordMetadata with extra_fields (from dataset store)
    extra = getattr(meta, "extra_fields", None)
    if isinstance(extra, dict):
        return str(extra.get(field, ""))
    return ""
