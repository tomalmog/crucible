"""Typed models for model versioning and registry.

This module defines immutable data models used by the model registry
to track model versions, tags, and branches.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelVersion:
    """Immutable record of a registered model version.

    Attributes:
        version_id: Unique identifier for this model version.
        model_path: Filesystem path to the model artifact.
        run_id: Optional training run that produced this model.
        tags: Descriptive tags associated with this version.
        created_at: ISO-8601 UTC creation timestamp.
        parent_version_id: Previous version this was derived from.
    """

    version_id: str
    model_path: str
    run_id: str | None
    tags: tuple[str, ...] = ()
    created_at: str = ""
    parent_version_id: str | None = None


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
class ModelBranch:
    """Named branch tracking the latest model version.

    Attributes:
        branch_name: Human-readable branch identifier.
        head_version_id: Model version at the branch head.
        created_at: ISO-8601 UTC creation timestamp.
    """

    branch_name: str
    head_version_id: str
    created_at: str
