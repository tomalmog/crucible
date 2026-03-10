"""Ingest orchestration pipeline.

This module coordinates source loading, transforms, checkpoints,
and dataset writes for resumable ingest workflows.
"""

from __future__ import annotations

import hashlib
import json

from core.config import CrucibleConfig
from core.logging_config import get_logger
from core.types import (
    DataRecord,
    DatasetWriteRequest,
    IngestOptions,
    RecordMetadata,
    SourceTextRecord,
)
from ingest.checkpoint_store import IngestCheckpointStore
from ingest.input_reader import read_source_records
from store.snapshot_store import DatasetStore
from transforms.exact_deduplication import build_record_id, remove_exact_duplicates
from transforms.language_detection import detect_languages
from transforms.quality_scoring import score_quality

_LOGGER = get_logger(__name__)


class IngestPipelineRunner:
    """Stateful runner for resumable ingest pipeline execution."""

    def __init__(self, options: IngestOptions, config: CrucibleConfig) -> None:
        self._options = options
        self._config = config
        self._store = DatasetStore(config)
        self._checkpoint = IngestCheckpointStore(config.data_root, options.dataset_name)
        run_signature = _build_run_signature(options)
        self._state = self._checkpoint.prepare_run(run_signature, options.resume)

    def run(self) -> str:
        """Execute ingest pipeline and return dataset name."""
        source_records = self._load_source_records()
        dedup_records = self._load_deduplicated_records(source_records)
        enriched_records = self._load_enriched_records(dedup_records)
        self._save_dataset(enriched_records)
        self._checkpoint.clear()
        _log_ingest_completion(
            self._options, len(source_records), len(enriched_records),
        )
        return self._options.dataset_name

    def _load_source_records(self) -> list[SourceTextRecord]:
        if self._checkpoint.has_stage(self._state, "source_loaded"):
            return self._checkpoint.load_source_records()
        source_records = read_source_records(self._options.source_uri, self._config)
        self._checkpoint.save_source_records(source_records)
        self._state = self._checkpoint.update_stage(self._state, "source_loaded")
        return source_records

    def _load_deduplicated_records(
        self,
        source_records: list[SourceTextRecord],
    ) -> list[SourceTextRecord]:
        if self._checkpoint.has_stage(self._state, "deduplicated"):
            return self._checkpoint.load_dedup_records()
        deduplicated_records = remove_exact_duplicates(source_records)
        self._checkpoint.save_dedup_records(deduplicated_records)
        self._state = self._checkpoint.update_stage(self._state, "deduplicated")
        return deduplicated_records

    def _load_enriched_records(self, dedup_records: list[SourceTextRecord]) -> list[DataRecord]:
        if self._checkpoint.has_stage(self._state, "enriched"):
            return self._checkpoint.load_enriched_records()
        enriched_records = _build_enriched_records(dedup_records, self._options.quality_model)
        self._checkpoint.save_enriched_records(enriched_records)
        self._state = self._checkpoint.update_stage(self._state, "enriched")
        return enriched_records

    def _save_dataset(self, records: list[DataRecord]) -> None:
        write_request = DatasetWriteRequest(
            dataset_name=self._options.dataset_name,
            records=tuple(records),
            source_uri=self._options.source_uri,
        )
        self._store.save_dataset(write_request)


def ingest_dataset(options: IngestOptions, config: CrucibleConfig) -> str:
    """Run the ingest pipeline and persist dataset records.

    Args:
        options: Ingest request options.
        config: Runtime configuration.

    Returns:
        Dataset name.

    Raises:
        CrucibleIngestError: If source read or transforms fail.
        CrucibleStoreError: If dataset persistence fails.
    """
    runner = IngestPipelineRunner(options, config)
    return runner.run()


def _build_enriched_records(
    source_records: list[SourceTextRecord],
    quality_model: str,
) -> list[DataRecord]:
    """Build fully-scored records from deduplicated input."""
    texts = [record.text for record in source_records]
    languages = detect_languages(texts)
    quality_results = score_quality(texts, quality_model)
    enriched_records: list[DataRecord] = []
    for source_record, language, quality_result in zip(source_records, languages, quality_results):
        metadata = RecordMetadata(
            source_uri=source_record.source_uri,
            language=language,
            quality_score=quality_result.quality_score,
            perplexity=quality_result.perplexity,
            extra_fields={"quality_model": quality_result.model_name},
        )
        enriched_records.append(
            DataRecord(
                record_id=build_record_id(source_record.text),
                text=source_record.text,
                metadata=metadata,
            )
        )
    return enriched_records


def _build_run_signature(options: IngestOptions) -> str:
    """Build deterministic run signature for checkpoint matching."""
    signature_payload = {
        "dataset_name": options.dataset_name,
        "source_uri": options.source_uri,
        "quality_model": options.quality_model,
    }
    serialized_payload = json.dumps(signature_payload, sort_keys=True)
    return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()


def _log_ingest_completion(
    options: IngestOptions,
    input_count: int,
    output_count: int,
) -> None:
    """Log pipeline completion with contextual metadata."""
    _LOGGER.info(
        "ingest_completed",
        dataset_name=options.dataset_name,
        source_uri=options.source_uri,
        input_count=input_count,
        output_count=output_count,
        quality_model=options.quality_model,
    )
