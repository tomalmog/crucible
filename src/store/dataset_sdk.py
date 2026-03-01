"""Python SDK for dataset operations.

This module exposes high-level APIs for ingest, loading, filtering,
and version inspection backed by the snapshot store.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from core.chat_types import ChatOptions, ChatResult
from core.config import ForgeConfig
from core.distillation_types import DistillationOptions
from core.domain_adaptation_types import DomainAdaptationOptions
from core.dpo_types import DpoOptions
from core.grpo_types import GrpoOptions
from core.kto_types import KtoOptions
from core.multimodal_types import MultimodalOptions
from core.orpo_types import OrpoOptions
from core.qlora_types import QloraOptions
from core.rlvr_types import RlvrOptions
from core.errors import (
    ForgeDistillationError, ForgeDpoError, ForgeGrpoError, ForgeKtoError,
    ForgeLoraError, ForgeMultimodalError, ForgeOrpoError, ForgeQloraError,
    ForgeRlhfError, ForgeRlvrError, ForgeSftError, ForgeServeError,
)
from core.lora_types import LoraTrainingOptions
from core.rlhf_types import RlhfOptions
from core.run_spec_execution import execute_run_spec_file
from core.sft_types import SftOptions
from core.types import (
    DataRecord,
    IngestOptions,
    MetadataFilter,
    SnapshotManifest,
    TrainingExportRequest,
    TrainingOptions,
    TrainingRunResult,
    VersionExportRequest,
)
from ingest.pipeline import ingest_dataset
from serve.chat_runner import run_chat
from serve.distillation_runner import run_distillation
from serve.domain_adaptation_runner import run_domain_adaptation
from serve.dpo_runner import run_dpo_training
from serve.grpo_runner import run_grpo_training
from serve.hardware_profile import detect_hardware_profile
from serve.kto_runner import run_kto_training
from serve.multimodal_runner import run_multimodal_training
from serve.orpo_runner import run_orpo_training
from serve.qlora_runner import run_qlora_training
from serve.rlhf_runner import run_rlhf_training
from serve.rlvr_runner import run_rlvr_training
from serve.lora_training_runner import run_lora_training
from serve.sft_runner import run_sft_training
from serve.training_run_registry import TrainingRunRegistry
from serve.training_run_types import TrainingRunRecord
from serve.training_runner import run_training
from store.snapshot_store import SnapshotStore


class ForgeClient:
    """Primary SDK entry point for phase-one workflows."""

    def __init__(self, config: ForgeConfig | None = None) -> None:
        self._config = config or ForgeConfig.from_env()
        self._store = SnapshotStore(self._config)

    def ingest(self, options: IngestOptions) -> str:
        """Ingest a source URI into a versioned dataset."""
        return ingest_dataset(options, self._config)

    def dataset(self, dataset_name: str) -> "Dataset":
        """Get dataset handle by name."""
        return Dataset(dataset_name, self._store)

    def resolve_dataset_source(self, dataset_name: str) -> str | None:
        """Look up the original source URI for a dataset's latest version."""
        if not dataset_name:
            return None
        try:
            manifests = self._store.list_versions(dataset_name)
        except Exception:
            return None
        return manifests[-1].source_uri if manifests else None

    def _resolve_data_path(self, options: object, data_path_field: str) -> object:
        """Fill an empty data path from the dataset's stored source URI.

        Raises:
            ForgeServeError: If the data path is empty and cannot be
                auto-resolved from the dataset's stored source URI.
        """
        current = getattr(options, data_path_field, "")
        if current:
            return options
        source = self.resolve_dataset_source(getattr(options, "dataset_name", ""))
        if source:
            resolved = Path(source).expanduser().resolve()
            if resolved.exists():
                return replace(options, **{data_path_field: str(resolved)})
        flag = data_path_field.replace("_", "-")
        dataset = getattr(options, "dataset_name", "")
        raise ForgeServeError(
            f"No data path provided (--{flag}). "
            f"Re-ingest the dataset{' ' + repr(dataset) if dataset else ''} "
            f"to store the source path, or pass --{flag} explicitly."
        )

    def train(self, options: TrainingOptions) -> TrainingRunResult:
        """Train a model on a dataset version using PyTorch."""
        return self.dataset(options.dataset_name).train(options)

    def sft_train(self, options: SftOptions) -> TrainingRunResult:
        """Run supervised fine-tuning on a dataset version."""
        options = self._resolve_data_path(options, "sft_data_path")
        if not options.dataset_name:
            return run_sft_training(
                records=[], options=options, random_seed=42,
                data_root=self._config.data_root, dataset_version_id="",
            )
        return self.dataset(options.dataset_name).sft_train(options)

    def lora_train(self, options: LoraTrainingOptions) -> TrainingRunResult:
        """Run LoRA fine-tuning."""
        options = self._resolve_data_path(options, "lora_data_path")
        if not options.dataset_name:
            from serve.lora_training_runner import run_lora_training
            return run_lora_training(
                options=options,
                random_seed=42,
                data_root=self._config.data_root,
                dataset_version_id="",
            )
        return self.dataset(options.dataset_name).lora_train(options)

    def dpo_train(self, options: DpoOptions) -> TrainingRunResult:
        """Run DPO preference optimization on a dataset version."""
        options = self._resolve_data_path(options, "dpo_data_path")
        if not options.dataset_name:
            return run_dpo_training(
                records=[], options=options, random_seed=42,
                data_root=self._config.data_root, dataset_version_id="",
            )
        return self.dataset(options.dataset_name).dpo_train(options)

    def rlhf_train(self, options: RlhfOptions) -> TrainingRunResult:
        """Run RLHF training with PPO on a dataset version."""
        if not options.dataset_name:
            return run_rlhf_training(
                records=[], options=options, random_seed=42,
                data_root=self._config.data_root, dataset_version_id="",
            )
        return self.dataset(options.dataset_name).rlhf_train(options)

    def distill(self, options: DistillationOptions) -> TrainingRunResult:
        """Run knowledge distillation on a dataset version."""
        return self.dataset(options.dataset_name).distill(options)

    def domain_adapt(self, options: DomainAdaptationOptions) -> TrainingRunResult:
        """Run domain adaptation on a dataset."""
        return self.dataset(options.dataset_name).domain_adapt(options)

    def grpo_train(self, options: GrpoOptions) -> TrainingRunResult:
        """Run GRPO training on a dataset version."""
        options = self._resolve_data_path(options, "grpo_data_path")
        if not options.dataset_name:
            return run_grpo_training(
                records=[], options=options, random_seed=42,
                data_root=self._config.data_root, dataset_version_id="",
            )
        return self.dataset(options.dataset_name).grpo_train(options)

    def qlora_train(self, options: QloraOptions) -> TrainingRunResult:
        """Run QLoRA training on a dataset version."""
        options = self._resolve_data_path(options, "qlora_data_path")
        if not options.dataset_name:
            return run_qlora_training(
                records=[], options=options, random_seed=42,
                data_root=self._config.data_root, dataset_version_id="",
            )
        return self.dataset(options.dataset_name).qlora_train(options)

    def kto_train(self, options: KtoOptions) -> TrainingRunResult:
        """Run KTO training on a dataset version."""
        options = self._resolve_data_path(options, "kto_data_path")
        if not options.dataset_name:
            return run_kto_training(
                records=[], options=options, random_seed=42,
                data_root=self._config.data_root, dataset_version_id="",
            )
        return self.dataset(options.dataset_name).kto_train(options)

    def orpo_train(self, options: OrpoOptions) -> TrainingRunResult:
        """Run ORPO training on a dataset version."""
        options = self._resolve_data_path(options, "orpo_data_path")
        if not options.dataset_name:
            return run_orpo_training(
                records=[], options=options, random_seed=42,
                data_root=self._config.data_root, dataset_version_id="",
            )
        return self.dataset(options.dataset_name).orpo_train(options)

    def multimodal_train(self, options: MultimodalOptions) -> TrainingRunResult:
        """Run multimodal fine-tuning on a dataset version."""
        options = self._resolve_data_path(options, "multimodal_data_path")
        if not options.dataset_name:
            return run_multimodal_training(
                records=[], options=options, random_seed=42,
                data_root=self._config.data_root, dataset_version_id="",
            )
        return self.dataset(options.dataset_name).multimodal_train(options)

    def rlvr_train(self, options: RlvrOptions) -> TrainingRunResult:
        """Run RLVR training on a dataset version."""
        options = self._resolve_data_path(options, "rlvr_data_path")
        if not options.dataset_name:
            return run_rlvr_training(
                records=[], options=options, random_seed=42,
                data_root=self._config.data_root, dataset_version_id="",
            )
        return self.dataset(options.dataset_name).rlvr_train(options)

    def chat(self, options: ChatOptions) -> ChatResult:
        """Run chat inference against a trained model."""
        if options.dataset_name is None:
            return run_chat(None, options)
        dataset = self.dataset(options.dataset_name)
        return dataset.chat(options)

    def with_data_root(self, data_root: str) -> "ForgeClient":
        """Clone the client with a different local data root."""
        resolved_root = Path(data_root).expanduser().resolve()
        return ForgeClient(replace(self._config, data_root=resolved_root))

    def run_spec(self, spec_file: str) -> tuple[str, ...]:
        """Execute a YAML run-spec through the shared execution engine."""
        return execute_run_spec_file(self, spec_file)

    def hardware_profile(self) -> dict[str, object]:
        """Detect local hardware profile and recommended defaults."""
        return detect_hardware_profile().to_dict()

    def list_training_runs(self) -> tuple[str, ...]:
        """List known training run IDs from lifecycle registry."""
        return TrainingRunRegistry(self._config.data_root).list_runs()

    def get_training_run(self, run_id: str) -> TrainingRunRecord:
        """Load one training run lifecycle record by ID."""
        return TrainingRunRegistry(self._config.data_root).load_run(run_id)

    def get_lineage_graph(self) -> dict[str, object]:
        """Load model/dataset lineage graph for this data root."""
        return TrainingRunRegistry(self._config.data_root).load_lineage_graph()

    def model_registry(self) -> "ModelRegistry":
        """Return a ModelRegistry for this data root."""
        from store.model_registry import ModelRegistry
        return ModelRegistry(self._config.data_root)


class Dataset:
    """SDK dataset handle for versioned records."""

    def __init__(self, dataset_name: str, store: SnapshotStore) -> None:
        self._dataset_name = dataset_name
        self._store = store

    @property
    def name(self) -> str:
        """Return dataset identifier."""
        return self._dataset_name

    def list_versions(self) -> list[SnapshotManifest]:
        """List all dataset versions."""
        return self._store.list_versions(self._dataset_name)

    def load_records(
        self, version_id: str | None = None,
    ) -> tuple[SnapshotManifest, list[DataRecord]]:
        """Load records for latest or target version."""
        return self._store.load_records(self._dataset_name, version_id)

    def filter(self, filter_spec: MetadataFilter) -> str:
        """Create a filtered snapshot from the latest version."""
        manifest = self._store.filter_records(self._dataset_name, filter_spec)
        return manifest.version_id

    def export(self, version_id: str, output_uri: str) -> None:
        """Export a snapshot version to an S3 destination."""
        request = VersionExportRequest(
            dataset_name=self._dataset_name,
            version_id=version_id,
            output_uri=output_uri,
        )
        self._store.export_version_to_s3(request)

    def export_training(
        self, output_dir: str, version_id: str | None = None,
        shard_size: int = 1000, include_metadata: bool = False,
    ) -> str:
        """Export snapshot into local sharded training files."""
        request = TrainingExportRequest(
            dataset_name=self._dataset_name,
            output_dir=output_dir,
            version_id=version_id,
            shard_size=shard_size,
            include_metadata=include_metadata,
        )
        manifest_path = self._store.export_training_data(request)
        return str(manifest_path)

    def train(self, options: TrainingOptions) -> TrainingRunResult:
        """Train a model on this dataset with default/custom loop."""
        if options.dataset_name != self._dataset_name:
            raise ForgeServeError(
                f"Training options dataset '{options.dataset_name}' does not match handle "
                f"'{self._dataset_name}'."
            )
        manifest, records = self.load_records(options.version_id)
        return run_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root, dataset_version_id=manifest.version_id,
        )

    def sft_train(self, options: SftOptions) -> TrainingRunResult:
        """Run supervised fine-tuning on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise ForgeSftError(
                f"SFT dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        manifest, records = self.load_records(options.version_id)
        return run_sft_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root, dataset_version_id=manifest.version_id,
        )

    def lora_train(self, options: LoraTrainingOptions) -> TrainingRunResult:
        """Run LoRA fine-tuning on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise ForgeLoraError(
                f"LoRA dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        manifest, _ = self.load_records(options.version_id)
        return run_lora_training(
            options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root, dataset_version_id=manifest.version_id,
        )

    def dpo_train(self, options: DpoOptions) -> TrainingRunResult:
        """Run DPO preference optimization on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise ForgeDpoError(
                f"DPO dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        manifest, records = self.load_records(options.version_id)
        return run_dpo_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root, dataset_version_id=manifest.version_id,
        )

    def rlhf_train(self, options: RlhfOptions) -> TrainingRunResult:
        """Run RLHF training with PPO on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise ForgeRlhfError(
                f"RLHF dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        manifest, records = self.load_records(options.version_id)
        return run_rlhf_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root, dataset_version_id=manifest.version_id,
        )

    def distill(self, options: DistillationOptions) -> TrainingRunResult:
        """Run knowledge distillation on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise ForgeDistillationError(
                f"Distillation dataset '{options.dataset_name}' != "
                f"handle '{self._dataset_name}'."
            )
        manifest, records = self.load_records(options.version_id)
        return run_distillation(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root, dataset_version_id=manifest.version_id,
        )

    def domain_adapt(self, options: DomainAdaptationOptions) -> TrainingRunResult:
        """Run domain adaptation on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise ForgeServeError(
                f"Domain adaptation dataset '{options.dataset_name}' != "
                f"handle '{self._dataset_name}'."
            )
        manifest, records = self.load_records(options.version_id)
        return run_domain_adaptation(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root, dataset_version_id=manifest.version_id,
        )

    def grpo_train(self, options: GrpoOptions) -> TrainingRunResult:
        """Run GRPO training on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise ForgeGrpoError(
                f"GRPO dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        manifest, records = self.load_records(options.version_id)
        return run_grpo_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root, dataset_version_id=manifest.version_id,
        )

    def qlora_train(self, options: QloraOptions) -> TrainingRunResult:
        """Run QLoRA training on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise ForgeQloraError(
                f"QLoRA dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        manifest, records = self.load_records(options.version_id)
        return run_qlora_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root, dataset_version_id=manifest.version_id,
        )

    def kto_train(self, options: KtoOptions) -> TrainingRunResult:
        """Run KTO training on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise ForgeKtoError(
                f"KTO dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        manifest, records = self.load_records(options.version_id)
        return run_kto_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root, dataset_version_id=manifest.version_id,
        )

    def orpo_train(self, options: OrpoOptions) -> TrainingRunResult:
        """Run ORPO training on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise ForgeOrpoError(
                f"ORPO dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        manifest, records = self.load_records(options.version_id)
        return run_orpo_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root, dataset_version_id=manifest.version_id,
        )

    def multimodal_train(self, options: MultimodalOptions) -> TrainingRunResult:
        """Run multimodal fine-tuning on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise ForgeMultimodalError(
                f"Multimodal dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        manifest, records = self.load_records(options.version_id)
        return run_multimodal_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root, dataset_version_id=manifest.version_id,
        )

    def rlvr_train(self, options: RlvrOptions) -> TrainingRunResult:
        """Run RLVR training on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise ForgeRlvrError(
                f"RLVR dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        manifest, records = self.load_records(options.version_id)
        return run_rlvr_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root, dataset_version_id=manifest.version_id,
        )

    def chat(self, options: ChatOptions) -> ChatResult:
        """Run one chat completion on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise ForgeServeError(
                f"Chat options dataset '{options.dataset_name}' does not match handle "
                f"'{self._dataset_name}'."
            )
        _, records = self.load_records(options.version_id)
        return run_chat(records, options)
