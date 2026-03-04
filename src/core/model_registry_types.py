"""Typed models for model versioning and registry.

This module defines immutable data models used by the model registry
to track model versions, tags, and groups.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelVersion:
    """Immutable record of a registered model version.

    Attributes:
        version_id: Unique identifier for this model version.
        model_name: Name of the model this version belongs to.
        model_path: Filesystem path to the model artifact.
        run_id: Optional training run that produced this model.
        tags: Descriptive tags associated with this version.
        created_at: ISO-8601 UTC creation timestamp.
        parent_version_id: Previous version this was derived from.
    """

    version_id: str
    model_name: str
    model_path: str
    run_id: str | None
    tags: tuple[str, ...] = ()
    created_at: str = ""
    parent_version_id: str | None = None
    location_type: str = "local"
    remote_host: str = ""
    remote_path: str = ""


@dataclass(frozen=True)
class ModelTag:
    """Named pointer to a specific model version.

    Attributes:
        tag_name: Human-readable tag identifier.
        version_id: Model version this tag points to.
        created_at: ISO-8601 UTC creation timestamp.
    """

    tag_name: str
    version_id: str
    created_at: str


@dataclass(frozen=True)
class DeleteResult:
    """Result of a model or version deletion operation.

    Attributes:
        versions_removed: Number of version records removed.
        local_paths_deleted: Paths successfully deleted from disk.
        local_paths_skipped: Paths skipped (outside safe zone).
        errors: Any errors encountered during deletion.
    """

    versions_removed: int
    local_paths_deleted: tuple[str, ...]
    local_paths_skipped: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ModelGroup:
    """Summary of a named model with its version history.

    Attributes:
        model_name: Human-readable model name.
        version_ids: Ordered tuple of version identifiers.
        active_version_id: Currently active version, or None.
        created_at: ISO-8601 UTC timestamp of first version.
    """

    model_name: str
    version_ids: tuple[str, ...]
    active_version_id: str | None
    created_at: str
