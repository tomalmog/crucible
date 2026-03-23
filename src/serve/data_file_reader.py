"""Unified reader for JSONL and Parquet training data files.

All training data loaders (SFT, DPO, KTO, GRPO, ORPO, RLVR,
multimodal) use this to read rows regardless of file format.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_data_rows(data_path: str) -> list[dict[str, object]]:
    """Read rows from a JSONL or Parquet file as dicts.

    Args:
        data_path: Path to a ``.jsonl`` or ``.parquet`` file.

    Returns:
        List of row dicts with string keys.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If pyarrow is needed but not installed.
    """
    path = Path(data_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix == ".parquet":
        return _read_parquet(path)
    return _read_jsonl(path)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read rows from a JSONL file."""
    rows: list[dict[str, object]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _read_parquet(path: Path) -> list[dict[str, object]]:
    """Read rows from a Parquet file."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Parquet support requires pyarrow. "
            "Install it to use .parquet training data."
        ) from exc
    table = pq.read_table(path)
    return table.to_pylist()
