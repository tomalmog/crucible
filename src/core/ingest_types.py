"""Ingest and store typed models.

This module defines immutable data models for ingest, filtering,
and export workflows used by CLI, SDK, and run-spec pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.constants import DEFAULT_QUALITY_MODEL

if TYPE_CHECKING:
    from core.types import DataRecord


@dataclass(frozen=True)
class IngestOptions:
    """Ingest command options.

    Attributes:
        dataset_name: Dataset name to create/update.
        source_uri: Input file path, directory, or S3 URI.
        output_uri: Optional output object-store URI for snapshot export.
        resume: Resume from the latest matching ingest checkpoint.
        incremental: Update an existing dataset from changed/new source records.
        quality_model: Quality scoring model identifier.
    """

    dataset_name: str
    source_uri: str
    output_uri: str | None = None
    resume: bool = False
    incremental: bool = False
    quality_model: str = DEFAULT_QUALITY_MODEL


@dataclass(frozen=True)
class SourceTextRecord:
    """Raw text input record before transforms.

    Attributes:
        source_uri: Source path or URI where text was loaded.
        text: Raw text content extracted from source.
    """

    source_uri: str
    text: str


@dataclass(frozen=True)
class MetadataFilter:
    """Metadata filter constraints for snapshot slicing.

    Attributes:
        language: Optional exact language match.
        min_quality_score: Optional lower quality threshold.
        source_prefix: Optional source URI prefix match.
    """

    language: str | None = None
    min_quality_score: float | None = None
    source_prefix: str | None = None


@dataclass(frozen=True)
class SnapshotWriteRequest:
    """Request payload for snapshot persistence.

    Attributes:
        dataset_name: Logical dataset identifier.
        records: Final transformed records to persist.
        recipe_steps: Ordered list of transform names.
        parent_version: Optional parent version id.
    """

    dataset_name: str
    records: tuple[DataRecord, ...]
    recipe_steps: tuple[str, ...]
    parent_version: str | None = None
    source_uri: str | None = None


@dataclass(frozen=True)
class VersionExportRequest:
    """Request payload for exporting a version.

    Attributes:
        dataset_name: Dataset identifier.
        version_id: Version to export.
        output_uri: Destination URI (currently s3:// only).
    """

    dataset_name: str
    version_id: str
    output_uri: str


@dataclass(frozen=True)
class TrainingExportRequest:
    """Request payload for training shard export.

    Attributes:
        dataset_name: Dataset identifier.
        version_id: Optional version id; latest if omitted.
        output_dir: Local output directory for shard files.
        shard_size: Number of records per shard file.
        include_metadata: Whether to include metadata in each JSONL row.
    """

    dataset_name: str
    output_dir: str
    version_id: str | None = None
    shard_size: int = 1000
    include_metadata: bool = False
