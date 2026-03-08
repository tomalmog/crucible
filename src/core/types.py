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
    DatasetWriteRequest as DatasetWriteRequest,
    IngestOptions as IngestOptions,
    MetadataFilter as MetadataFilter,
    SourceTextRecord as SourceTextRecord,
    TrainingExportRequest as TrainingExportRequest,
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
class DatasetManifest:
    """Dataset metadata persisted alongside records.

    Attributes:
        dataset_name: Logical dataset identifier.
        created_at: UTC creation timestamp.
        record_count: Number of records in dataset.
        source_uri: Original ingest source path, if available.
    """

    dataset_name: str
    created_at: datetime
    record_count: int
    source_uri: str | None = None
