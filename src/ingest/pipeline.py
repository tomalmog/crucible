"""Ingest orchestration for phase-one pipelines.

This module coordinates source loading, transforms, checkpoints,
and snapshot writes for resumable and incremental ingest workflows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from core.config import ForgeConfig
from core.errors import ForgeStoreError
from core.logging_config import get_logger
from core.types import (
    DataRecord,
    IngestOptions,
    RecordMetadata,
    SnapshotManifest,
    SnapshotWriteRequest,
    SourceTextRecord,
    VersionExportRequest,
)
from ingest.checkpoint_store import IngestCheckpointStore
from ingest.incremental_ingest import (
    IncrementalSelection,
    merge_incremental_records,
    select_incremental_records,
)
from ingest.input_reader import read_source_records
from store.snapshot_store import SnapshotStore
from transforms.exact_deduplication import build_record_id, remove_exact_duplicates
from transforms.language_detection import detect_languages
from transforms.quality_scoring import score_quality

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class IncrementalContext:
    """Incremental ingest context for downstream stages."""

    selection: IncrementalSelection
    parent_version: str | None


class IngestPipelineRunner:
    """Stateful runner for resumable ingest pipeline execution."""

    def __init__(self, options: IngestOptions, config: ForgeConfig) -> None:
        self._options = options
        self._config = config
        self._store = SnapshotStore(config)
        self._checkpoint = IngestCheckpointStore(config.data_root, options.dataset_name)
        run_signature = _build_run_signature(options)
        self._state = self._checkpoint.prepare_run(run_signature, options.resume)

    def run(self) -> str:
        """Execute ingest pipeline and return created version id."""
        source_records = self._load_source_records()
        incremental_context = self._load_incremental_context(source_records)
        dedup_records = self._load_deduplicated_records(
            incremental_context.selection.records_to_process
        )
        enriched_records = self._load_enriched_records(dedup_records)
        final_records = _build_final_records(incremental_context.selection, enriched_records)
        manifest = self._create_snapshot(final_records, incremental_context.parent_version)
        self._export_if_requested(manifest.version_id)
        self._checkpoint.clear()
        _log_ingest_completion(
            self._options, len(source_records), len(final_records), manifest.version_id
        )
        return manifest.version_id

    def _load_source_records(self) -> list[SourceTextRecord]:
        if self._checkpoint.has_stage(self._state, "source_loaded"):
            return self._checkpoint.load_source_records()
        source_records = read_source_records(self._options.source_uri, self._config)
        self._checkpoint.save_source_records(source_records)
        self._state = self._checkpoint.update_stage(self._state, "source_loaded", None)
        return source_records

    def _load_incremental_context(
        self,
        source_records: list[SourceTextRecord],
    ) -> IncrementalContext:
        if self._checkpoint.has_stage(self._state, "incremental_selected"):
            selection = IncrementalSelection(
                records_to_process=self._checkpoint.load_work_records(),
                unchanged_records=self._checkpoint.load_unchanged_records(),
            )
            return IncrementalContext(
                selection=selection, parent_version=self._state.parent_version
            )
        context = self._compute_incremental_context(source_records)
        self._checkpoint.save_work_records(context.selection.records_to_process)
        self._checkpoint.save_unchanged_records(context.selection.unchanged_records)
        self._state = self._checkpoint.update_stage(
            self._state,
            "incremental_selected",
            context.parent_version,
        )
        return context

    def _compute_incremental_context(
        self,
        source_records: list[SourceTextRecord],
    ) -> IncrementalContext:
        if not self._options.incremental:
            selection = IncrementalSelection(
                records_to_process=source_records, unchanged_records=[]
            )
            return IncrementalContext(selection=selection, parent_version=None)
        latest_snapshot = _load_latest_snapshot(self._store, self._options.dataset_name)
        if latest_snapshot is None:
            selection = IncrementalSelection(
                records_to_process=source_records, unchanged_records=[]
            )
            return IncrementalContext(selection=selection, parent_version=None)
        manifest, existing_records = latest_snapshot
        selection = select_incremental_records(source_records, existing_records)
        return IncrementalContext(selection=selection, parent_version=manifest.version_id)

    def _load_deduplicated_records(
        self,
        work_records: list[SourceTextRecord],
    ) -> list[SourceTextRecord]:
        if self._checkpoint.has_stage(self._state, "deduplicated"):
            return self._checkpoint.load_dedup_records()
        deduplicated_records = remove_exact_duplicates(work_records)
        self._checkpoint.save_dedup_records(deduplicated_records)
        self._state = self._checkpoint.update_stage(
            self._state,
            "deduplicated",
            self._state.parent_version,
        )
        return deduplicated_records

    def _load_enriched_records(self, dedup_records: list[SourceTextRecord]) -> list[DataRecord]:
        if self._checkpoint.has_stage(self._state, "enriched"):
            return self._checkpoint.load_enriched_records()
        enriched_records = _build_enriched_records(dedup_records, self._options.quality_model)
        self._checkpoint.save_enriched_records(enriched_records)
        self._state = self._checkpoint.update_stage(
            self._state,
            "enriched",
            self._state.parent_version,
        )
        return enriched_records

    def _create_snapshot(
        self,
        records: list[DataRecord],
        parent_version: str | None,
    ) -> SnapshotManifest:
        recipe_steps = _build_recipe_steps(self._options.incremental, self._options.quality_model)
        write_request = SnapshotWriteRequest(
            dataset_name=self._options.dataset_name,
            records=tuple(records),
            recipe_steps=recipe_steps,
            parent_version=parent_version,
            source_uri=self._options.source_uri,
        )
        return self._store.create_snapshot(write_request)

    def _export_if_requested(self, version_id: str) -> None:
        if not self._options.output_uri:
            return
        export_request = VersionExportRequest(
            dataset_name=self._options.dataset_name,
            version_id=version_id,
            output_uri=self._options.output_uri,
        )
        self._store.export_version_to_s3(export_request)


def ingest_dataset(options: IngestOptions, config: ForgeConfig) -> str:
    """Run the phase-one ingest pipeline and persist a snapshot.

    Args:
        options: Ingest request options.
        config: Runtime configuration.

    Returns:
        Created snapshot version id.

    Raises:
        ForgeIngestError: If source read or transforms fail.
        ForgeStoreError: If snapshot persistence fails.
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


def _load_latest_snapshot(
    store: SnapshotStore,
    dataset_name: str,
) -> tuple[SnapshotManifest, list[DataRecord]] | None:
    """Load latest snapshot if dataset exists, else return None."""
    try:
        return store.load_records(dataset_name)
    except ForgeStoreError as error:
        if "catalog not found" not in str(error):
            raise
        return None


def _build_final_records(
    selection: IncrementalSelection,
    enriched_records: list[DataRecord],
) -> list[DataRecord]:
    """Build final snapshot records from incremental selection."""
    return merge_incremental_records(selection.unchanged_records, enriched_records)


def _build_run_signature(options: IngestOptions) -> str:
    """Build deterministic run signature for checkpoint matching."""
    signature_payload = {
        "dataset_name": options.dataset_name,
        "source_uri": options.source_uri,
        "incremental": options.incremental,
        "quality_model": options.quality_model,
    }
    serialized_payload = json.dumps(signature_payload, sort_keys=True)
    return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()


def _build_recipe_steps(incremental: bool, quality_model: str) -> tuple[str, ...]:
    """Build ordered recipe steps for snapshot lineage."""
    steps: list[str] = [
        "exact_deduplication",
        "language_detection",
        f"quality_scoring:{quality_model}",
    ]
    if incremental:
        steps.insert(0, "incremental_selection")
    return tuple(steps)


def _log_ingest_completion(
    options: IngestOptions,
    input_count: int,
    output_count: int,
    version_id: str,
) -> None:
    """Log pipeline completion with contextual metadata."""
    _LOGGER.info(
        "ingest_completed",
        dataset_name=options.dataset_name,
        source_uri=options.source_uri,
        input_count=input_count,
        output_count=output_count,
        version_id=version_id,
        output_uri=options.output_uri,
        incremental=options.incremental,
        quality_model=options.quality_model,
    )
