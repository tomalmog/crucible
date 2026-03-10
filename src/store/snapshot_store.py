"""Dataset store for flat record layout.

This module persists datasets as flat directories with records and manifest.
It provides load and export operations for the SDK.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from core.config import CrucibleConfig
from core.constants import (
    DATASETS_DIR_NAME,
    MANIFEST_FILE_NAME,
    RECORDS_FILE_NAME,
)
from core.errors import CrucibleStoreError
from core.logging_config import get_logger
from core.types import (
    DataRecord,
    DatasetManifest,
    DatasetWriteRequest,
    TrainingExportRequest,
)
from store.catalog_io import read_manifest_file, write_manifest_file
from store.lance_dataset import read_version_payload, write_version_payload
from store.training_export import export_training_shards

_LOGGER = get_logger(__name__)


class DatasetStore:
    """Flat dataset store implementation.

    Each dataset lives at datasets/{name}/ with records.jsonl and manifest.json.
    """

    def __init__(self, config: CrucibleConfig) -> None:
        self._config = config
        self._datasets_root = config.data_root / DATASETS_DIR_NAME
        self._datasets_root.mkdir(parents=True, exist_ok=True)

    @property
    def random_seed(self) -> int:
        """Return configured random seed for deterministic operations."""
        return self._config.random_seed

    @property
    def data_root(self) -> Path:
        """Return resolved data root used by this store."""
        return self._config.data_root

    def save_dataset(self, request: DatasetWriteRequest) -> DatasetManifest:
        """Save dataset records, overwriting any previous data.

        Args:
            request: Dataset write request payload.

        Returns:
            Persisted dataset manifest.
        """
        dataset_dir = self._dataset_dir(request.dataset_name)
        lance_written = write_version_payload(dataset_dir, list(request.records))
        manifest = DatasetManifest(
            dataset_name=request.dataset_name,
            created_at=datetime.now(timezone.utc),
            record_count=len(request.records),
            source_uri=request.source_uri,
        )
        write_manifest_file(dataset_dir, manifest, lance_written)
        _LOGGER.info(
            "dataset_saved",
            dataset_name=request.dataset_name,
            record_count=manifest.record_count,
            lance_written=lance_written,
        )
        return manifest

    def load_records(
        self,
        dataset_name: str,
    ) -> tuple[DatasetManifest, list[DataRecord]]:
        """Load records for a dataset.

        Args:
            dataset_name: Dataset identifier.

        Returns:
            Pair of manifest and loaded records.

        Raises:
            CrucibleStoreError: If dataset is missing.
        """
        dataset_dir = self._dataset_dir(dataset_name)
        manifest_path = dataset_dir / MANIFEST_FILE_NAME
        if not manifest_path.exists():
            raise CrucibleStoreError(
                f"Dataset '{dataset_name}' not found (no manifest at {manifest_path}). "
                "Ingest data before reading."
            )
        manifest = read_manifest_file(manifest_path)
        records = read_version_payload(dataset_dir)
        return manifest, records

    def export_training_data(self, request: TrainingExportRequest) -> Path:
        """Export a dataset into sharded local training files.

        Args:
            request: Training export request.

        Returns:
            Path to generated training manifest.
        """
        manifest, records = self.load_records(request.dataset_name)
        manifest_path = export_training_shards(request, manifest, records)
        _LOGGER.info(
            "training_export_completed",
            dataset_name=request.dataset_name,
            output_dir=request.output_dir,
            shard_size=request.shard_size,
            include_metadata=request.include_metadata,
        )
        return manifest_path

    def _dataset_dir(self, dataset_name: str) -> Path:
        """Return dataset directory path, creating it if needed."""
        dataset_dir = self._datasets_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir


# Keep old class name as alias for imports that haven't been updated yet
SnapshotStore = DatasetStore
