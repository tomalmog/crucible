"""JSON I/O helpers for training lifecycle and lineage metadata."""

from __future__ import annotations

import json
import os
import tempfile
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
    """Atomically write one JSON payload to disk.

    Writes to a temporary file in the same directory, then uses
    ``os.replace`` to atomically swap it into place. This prevents
    partial writes from corrupting the file if the process is
    interrupted mid-write.
    """
    json_str = json.dumps(payload, indent=2) + "\n"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    fd = -1
    tmp_path = ""
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(payload_path.parent), suffix=".tmp",
        )
        os.write(fd, json_str.encode("utf-8"))
        os.close(fd)
        fd = -1
        os.replace(tmp_path, str(payload_path))
    except OSError as error:
        if fd >= 0:
            os.close(fd)
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise CrucibleServeError(
            f"Failed to write metadata file {payload_path}: {error}."
        ) from error
