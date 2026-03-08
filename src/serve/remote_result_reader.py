"""Read result.json from remote Slurm job working directories.

Provides shared helpers for reading and parsing result.json
from remote clusters, used by state sync and model puller modules.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.slurm_types import RemoteJobRecord
    from serve.ssh_connection import SshSession


def read_remote_result(
    session: SshSession,
    record: RemoteJobRecord,
) -> dict[str, str]:
    """Read and parse result.json from the remote job's output dir.

    Returns an empty dict if the file is missing or unparseable.
    """
    result_path = f"{record.remote_output_dir}/result.json"
    stdout, _, code = session.execute(f"cat '{result_path}'", timeout=15)
    if code != 0:
        return {}
    try:
        return dict(json.loads(stdout.strip()))
    except (json.JSONDecodeError, ValueError):
        return {}


def extract_result_error(
    session: SshSession,
    record: RemoteJobRecord,
) -> str:
    """Return the error message from result.json, or empty string."""
    result = read_remote_result(session, record)
    if result.get("status") == "failed":
        return str(result.get("error", ""))[:300]
    return ""


def extract_result_model_path(
    session: SshSession,
    record: RemoteJobRecord,
) -> str:
    """Return the model_path from result.json, or empty string."""
    result = read_remote_result(session, record)
    return str(result.get("model_path", ""))
