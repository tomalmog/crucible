"""Shared typed models.

This module defines immutable data models used by ingest, store,
SDK, and serving layers to keep interfaces explicit and stable.

Core record types live here. Training and ingest types are defined in
their respective modules and re-exported for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping

from core.ingest_types import (
    IngestOptions as IngestOptions,
    MetadataFilter as MetadataFilter,
    SnapshotWriteRequest as SnapshotWriteRequest,
    SourceTextRecord as SourceTextRecord,
    TrainingExportRequest as TrainingExportRequest,
    VersionExportRequest as VersionExportRequest,
)
from core.training_types import (
    BatchLossMetric as BatchLossMetric,
    DataLoaderOptions as DataLoaderOptions,
    EpochMetric as EpochMetric,
    OptimizerType as OptimizerType,
    PositionEmbeddingType as PositionEmbeddingType,
    PrecisionMode as PrecisionMode,
    SchedulerType as SchedulerType,
    TrainingOptions as TrainingOptions,
    TrainingRunResult as TrainingRunResult,
)


@dataclass(frozen=True)
class RecordMetadata:
    """Metadata attached to each training record.

    Attributes:
        source_uri: Origin path or URI of the record.
        language: Detected language code.
        quality_score: Normalized quality score in [0, 1].
        perplexity: Perplexity value used to derive quality score.
        extra_fields: User-extensible metadata dictionary.
    """

    source_uri: str
    language: str
    quality_score: float
    perplexity: float
    extra_fields: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DataRecord:
    """Canonical text training record.

    Attributes:
        record_id: Stable content hash identifier.
        text: Raw text payload.
        metadata: Typed metadata fields.
    """

    record_id: str
    text: str
    metadata: RecordMetadata


@dataclass(frozen=True)
class SnapshotManifest:
    """Immutable snapshot metadata for versioning.

    Attributes:
        dataset_name: Logical dataset identifier.
        version_id: Immutable snapshot id.
        created_at: UTC creation timestamp.
        parent_version: Previous version id when derived.
        recipe_steps: Ordered transforms used to create snapshot.
        record_count: Number of records in snapshot.
    """

    dataset_name: str
    version_id: str
    created_at: datetime
    parent_version: str | None
    recipe_steps: tuple[str, ...]
    record_count: int
