"""Snapshot store and metadata catalog.

This module persists immutable dataset versions with lineage metadata.
It provides load, list, filter, and export operations for the SDK.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from core.config import ForgeConfig
from core.constants import (
    CATALOG_FILE_NAME,
    DATASETS_DIR_NAME,
    VERSIONS_DIR_NAME,
)
from core.errors import ForgeStoreError
from core.logging_config import get_logger
from core.s3_uri import parse_s3_uri
from core.types import (
    DataRecord,
    MetadataFilter,
    SnapshotManifest,
    SnapshotWriteRequest,
    TrainingExportRequest,
    VersionExportRequest,
)
from store.catalog_io import (
    build_version_id,
    manifest_from_dict,
    read_catalog_file,
    update_catalog,
    write_manifest_file,
)
from store.lance_dataset import read_version_payload, write_version_payload
from store.metadata_filtering import filter_records
from store.s3_export import create_s3_client, upload_directory
from store.training_export import export_training_shards

_LOGGER = get_logger(__name__)


class SnapshotStore:
    """Immutable snapshot store implementation.

    This class owns dataset directories, version manifests,
    and metadata catalog updates for phase-one operations.
    """

    def __init__(self, config: ForgeConfig) -> None:
        """Initialize snapshot store from config.

        Args:
            config: Runtime configuration.
        """
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

    def create_snapshot(self, request: SnapshotWriteRequest) -> SnapshotManifest:
        """Create a new immutable dataset snapshot.

        Args:
            request: Snapshot write request payload.

        Returns:
            Persisted snapshot manifest.

        Raises:
            ForgeStoreError: If persistence fails.
        """
        dataset_root = self._dataset_root(request.dataset_name)
        version_id = build_version_id(request.dataset_name, request.records)
        version_dir = dataset_root / VERSIONS_DIR_NAME / version_id
        version_dir.mkdir(parents=True, exist_ok=False)
        lance_written = write_version_payload(version_dir, list(request.records))
        manifest = SnapshotManifest(
            dataset_name=request.dataset_name,
            version_id=version_id,
            created_at=datetime.now(timezone.utc),
            parent_version=request.parent_version,
            recipe_steps=request.recipe_steps,
            record_count=len(request.records),
            source_uri=request.source_uri,
        )
        write_manifest_file(version_dir, manifest, lance_written)
        update_catalog(dataset_root / CATALOG_FILE_NAME, manifest)
        _LOGGER.info(
            "snapshot_created",
            dataset_name=request.dataset_name,
            version_id=version_id,
            record_count=manifest.record_count,
            parent_version=manifest.parent_version,
            lance_written=lance_written,
        )
        return manifest

    def list_versions(self, dataset_name: str) -> list[SnapshotManifest]:
        """List manifests for a dataset sorted by creation time.

        Args:
            dataset_name: Dataset identifier.

        Returns:
            Ordered manifest list.

        Raises:
            ForgeStoreError: If dataset catalog does not exist.
        """
        catalog_path = self._dataset_root(dataset_name) / CATALOG_FILE_NAME
        catalog = read_catalog_file(catalog_path)
        version_payloads = cast(list[dict[str, Any]], catalog["versions"])
        versions = [manifest_from_dict(item) for item in version_payloads]
        return sorted(versions, key=lambda item: item.created_at)

    def load_records(
        self,
        dataset_name: str,
        version_id: str | None = None,
    ) -> tuple[SnapshotManifest, list[DataRecord]]:
        """Load records for a dataset snapshot.

        Args:
            dataset_name: Dataset identifier.
            version_id: Optional snapshot version; latest when omitted.

        Returns:
            Pair of manifest and loaded records.

        Raises:
            ForgeStoreError: If dataset/version is missing.
        """
        manifest = self._resolve_manifest(dataset_name, version_id)
        version_dir = self._version_dir(dataset_name, manifest.version_id)
        records = read_version_payload(version_dir)
        return manifest, records

    def filter_records(
        self,
        dataset_name: str,
        filter_spec: MetadataFilter,
    ) -> SnapshotManifest:
        """Create a filtered immutable snapshot from latest version.

        Args:
            dataset_name: Dataset to filter.
            filter_spec: Metadata filter criteria.

        Returns:
            New filtered snapshot manifest.
        """
        parent_manifest, records = self.load_records(dataset_name)
        filtered_records = filter_records(records, filter_spec)
        recipe_steps = parent_manifest.recipe_steps + ("metadata_filter",)
        request = SnapshotWriteRequest(
            dataset_name=dataset_name,
            records=tuple(filtered_records),
            recipe_steps=recipe_steps,
            parent_version=parent_manifest.version_id,
        )
        return self.create_snapshot(request)

    def export_version_to_s3(self, request: VersionExportRequest) -> None:
        """Export a version directory to S3.

        Args:
            request: Export request payload.

        Raises:
            ForgeStoreError: If export fails.
        """
        location = parse_s3_uri(request.output_uri, domain="store")
        version_dir = self._version_dir(request.dataset_name, request.version_id)
        s3_client = create_s3_client(self._config)
        upload_directory(s3_client, version_dir, location.bucket, location.prefix)
        _LOGGER.info(
            "snapshot_exported",
            dataset_name=request.dataset_name,
            version_id=request.version_id,
            output_uri=request.output_uri,
        )

    def export_training_data(self, request: TrainingExportRequest) -> Path:
        """Export a snapshot into sharded local training files.

        Args:
            request: Training export request.

        Returns:
            Path to generated training manifest.
        """
        manifest, records = self.load_records(request.dataset_name, request.version_id)
        manifest_path = export_training_shards(request, manifest, records)
        _LOGGER.info(
            "training_export_completed",
            dataset_name=request.dataset_name,
            version_id=manifest.version_id,
            output_dir=request.output_dir,
            shard_size=request.shard_size,
            include_metadata=request.include_metadata,
        )
        return manifest_path

    def _dataset_root(self, dataset_name: str) -> Path:
        """Return dataset root path and ensure base directories.

        Args:
            dataset_name: Dataset identifier.

        Returns:
            Dataset root path.
        """
        dataset_root = self._datasets_root / dataset_name
        (dataset_root / VERSIONS_DIR_NAME).mkdir(parents=True, exist_ok=True)
        return dataset_root

    def _resolve_manifest(self, dataset_name: str, version_id: str | None) -> SnapshotManifest:
        """Resolve a target manifest.

        Args:
            dataset_name: Dataset identifier.
            version_id: Optional version id.

        Returns:
            Resolved snapshot manifest.

        Raises:
            ForgeStoreError: If catalog or target version is missing.
        """
        manifests = self.list_versions(dataset_name)
        if not manifests:
            raise ForgeStoreError(
                f"No versions exist for dataset '{dataset_name}'. "
                "Ingest data before reading snapshots."
            )
        if version_id is None:
            return manifests[-1]
        for manifest in manifests:
            if manifest.version_id == version_id:
                return manifest
        raise ForgeStoreError(
            f"Version '{version_id}' not found for dataset '{dataset_name}'. "
            "Use list_versions to discover valid version ids."
        )

    def _version_dir(self, dataset_name: str, version_id: str) -> Path:
        """Return snapshot version directory.

        Args:
            dataset_name: Dataset identifier.
            version_id: Snapshot version id.

        Returns:
            Version directory path.

        Raises:
            ForgeStoreError: If version directory is missing.
        """
        version_dir = self._dataset_root(dataset_name) / VERSIONS_DIR_NAME / version_id
        if not version_dir.exists():
            raise ForgeStoreError(
                f"Missing snapshot directory for {dataset_name}:{version_id} at {version_dir}. "
                "Recreate the snapshot before loading or exporting."
            )
        return version_dir
