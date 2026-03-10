"""JSON I/O helpers for training lifecycle and lineage metadata."""

from __future__ import annotations

import json
from pathlib import Path

from core.errors import CrucibleServeError


def read_json_file(payload_path: Path, default_value: object | None = None) -> object:
    """Read JSON payload from disk with optional default when missing."""
    if default_value is not None and not payload_path.exists():
        return default_value
    try:
        return json.loads(payload_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CrucibleServeError(
            f"Missing required run metadata at {payload_path}. Run may be incomplete."
        ) from error
    except json.JSONDecodeError as error:
        raise CrucibleServeError(f"Failed to parse JSON at {payload_path}: {error.msg}.") from error
    except OSError as error:
        raise CrucibleServeError(f"Failed to read metadata file {payload_path}: {error}.") from error


def write_json_file(payload_path: Path, payload: object) -> None:
    """Write one JSON payload to disk with traceable errors."""
    try:
        payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        raise CrucibleServeError(f"Failed to write metadata file {payload_path}: {error}.") from error
