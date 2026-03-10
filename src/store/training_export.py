"""Training export helpers for dataset records.

This module converts dataset records into sharded JSONL outputs.
It writes a manifest for downstream training pipeline consumption.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from core.errors import CrucibleStoreError
from core.types import DataRecord, DatasetManifest, TrainingExportRequest
from store.record_payload import data_record_to_payload


def export_training_shards(
    request: TrainingExportRequest,
    manifest: DatasetManifest,
    records: list[DataRecord],
) -> Path:
    """Export records into local training shards and manifest.

    Args:
        request: Training export request options.
        manifest: Source dataset manifest.
        records: Dataset records.

    Returns:
        Path to generated export manifest file.

    Raises:
        CrucibleStoreError: If export options are invalid.
    """
    _validate_shard_size(request.shard_size)
    export_dir = _build_export_dir(request.output_dir, manifest)
    shard_paths = _write_shards(export_dir, records, request)
    manifest_path = _write_export_manifest(export_dir, manifest, records, shard_paths)
    return manifest_path


def _validate_shard_size(shard_size: int) -> None:
    """Validate shard size input."""
    if shard_size < 1:
        raise CrucibleStoreError(
            f"Invalid shard size {shard_size}: expected value >= 1. "
            "Use --shard-size with a positive integer."
        )


def _build_export_dir(output_dir: str, manifest: DatasetManifest) -> Path:
    """Build and create destination export directory."""
    base_dir = Path(output_dir).expanduser().resolve()
    export_dir = base_dir / manifest.dataset_name
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def _write_shards(
    export_dir: Path,
    records: list[DataRecord],
    request: TrainingExportRequest,
) -> list[Path]:
    """Write records to shard JSONL files."""
    shard_paths: list[Path] = []
    for offset in range(0, len(records), request.shard_size):
        shard_records = records[offset : offset + request.shard_size]
        shard_index = offset // request.shard_size
        shard_path = export_dir / f"shard-{shard_index:05d}.jsonl"
        _write_shard_file(shard_path, shard_records, request.include_metadata)
        shard_paths.append(shard_path)
    return shard_paths


def _write_shard_file(
    shard_path: Path,
    shard_records: list[DataRecord],
    include_metadata: bool,
) -> None:
    """Write one shard file."""
    lines = [_serialize_training_row(record, include_metadata) for record in shard_records]
    shard_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _serialize_training_row(record: DataRecord, include_metadata: bool) -> str:
    """Serialize one training row."""
    if include_metadata:
        payload = data_record_to_payload(record)
    else:
        payload = {"text": record.text}
    return json.dumps(payload, sort_keys=True)


def _write_export_manifest(
    export_dir: Path,
    source_manifest: DatasetManifest,
    records: list[DataRecord],
    shard_paths: list[Path],
) -> Path:
    """Write training export manifest file."""
    manifest_payload = {
        "dataset_name": source_manifest.dataset_name,
        "record_count": len(records),
        "shard_count": len(shard_paths),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "shards": [path.name for path in shard_paths],
    }
    manifest_path = export_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path
