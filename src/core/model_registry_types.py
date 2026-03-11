"""Typed models for model registry.

This module defines immutable data models used by the model registry
to track registered models.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEntry:
    """Immutable record of a registered model.

    Attributes:
        model_name: Human-readable model name (unique key).
        model_path: Filesystem path to the model artifact.
        run_id: Optional training run that produced this model.
        created_at: ISO-8601 UTC creation timestamp.
        location_type: One of "local", "remote", or "both".
        remote_host: Hostname of remote cluster (if remote).
        remote_path: Path on remote cluster (if remote).
    """

    model_name: str
    model_path: str
    run_id: str | None = None
    created_at: str = ""
    location_type: str = "local"
    remote_host: str = ""
    remote_path: str = ""


@dataclass(frozen=True)
class DeleteResult:
    """Result of a model deletion operation.

    Attributes:
        entries_removed: Number of model entries removed.
        local_paths_deleted: Paths successfully deleted from disk.
        local_paths_skipped: Paths skipped (outside safe zone).
        errors: Any errors encountered during deletion.
    """

    entries_removed: int
    local_paths_deleted: tuple[str, ...]
    local_paths_skipped: tuple[str, ...]
    errors: tuple[str, ...]
