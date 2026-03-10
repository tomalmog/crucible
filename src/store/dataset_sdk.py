"""Python SDK for dataset operations.

This module exposes high-level APIs for ingest, loading,
and training backed by the dataset store.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from core.chat_types import ChatOptions, ChatResult
from core.config import CrucibleConfig
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
    CrucibleDistillationError, CrucibleDpoError, CrucibleGrpoError, CrucibleKtoError,
    CrucibleLoraError, CrucibleMultimodalError, CrucibleOrpoError, CrucibleQloraError,
    CrucibleRlhfError, CrucibleRlvrError, CrucibleSftError, CrucibleServeError,
)
from core.lora_types import LoraTrainingOptions
from core.rlhf_types import RlhfOptions
from core.run_spec_execution import execute_run_spec_file
from core.sft_types import SftOptions
from core.types import (
    DataRecord,
    DatasetManifest,
    IngestOptions,
    TrainingExportRequest,
    TrainingOptions,
    TrainingRunResult,
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
from store.snapshot_store import DatasetStore


class CrucibleClient:
    """Primary SDK entry point for dataset workflows."""

    def __init__(self, config: CrucibleConfig | None = None) -> None:
        self._config = config or CrucibleConfig.from_env()
        self._store = DatasetStore(self._config)

    def ingest(self, options: IngestOptions) -> str:
        """Ingest a source URI into a dataset."""
        return ingest_dataset(options, self._config)

    def dataset(self, dataset_name: str) -> "Dataset":
        """Get dataset handle by name."""
        return Dataset(dataset_name, self._store)

    def resolve_dataset_source(self, dataset_name: str) -> str | None:
        """Look up the original source URI for a dataset."""
        if not dataset_name:
            return None
        try:
            manifest, _ = self._store.load_records(dataset_name)
        except Exception:
            return None
        return manifest.source_uri

    def _resolve_data_path(self, options: object, data_path_field: str) -> object:
        """Fill an empty data path from the dataset's stored source URI."""
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
        raise CrucibleServeError(
            f"No data path provided (--{flag}). "
            f"Re-ingest the dataset{' ' + repr(dataset) if dataset else ''} "
            f"to store the source path, or pass --{flag} explicitly."
        )

    def train(self, options: TrainingOptions) -> TrainingRunResult:
        """Train a model on a dataset using PyTorch."""
        return self.dataset(options.dataset_name).train(options)

    def sft_train(self, options: SftOptions) -> TrainingRunResult:
        """Run supervised fine-tuning."""
        options = self._resolve_data_path(options, "sft_data_path")
        if not options.dataset_name:
            return run_sft_training(
                records=[], options=options, random_seed=42,
                data_root=self._config.data_root,
            )
        return self.dataset(options.dataset_name).sft_train(options)

    def lora_train(self, options: LoraTrainingOptions) -> TrainingRunResult:
        """Run LoRA fine-tuning."""
        options = self._resolve_data_path(options, "lora_data_path")
        if not options.dataset_name:
            return run_lora_training(
                options=options, random_seed=42,
                data_root=self._config.data_root,
            )
        return self.dataset(options.dataset_name).lora_train(options)

    def dpo_train(self, options: DpoOptions) -> TrainingRunResult:
        """Run DPO preference optimization."""
        options = self._resolve_data_path(options, "dpo_data_path")
        if not options.dataset_name:
            return run_dpo_training(
                records=[], options=options, random_seed=42,
                data_root=self._config.data_root,
            )
        return self.dataset(options.dataset_name).dpo_train(options)

    def rlhf_train(self, options: RlhfOptions) -> TrainingRunResult:
        """Run RLHF training with PPO."""
        if not options.dataset_name:
            return run_rlhf_training(
                records=[], options=options, random_seed=42,
                data_root=self._config.data_root,
            )
        return self.dataset(options.dataset_name).rlhf_train(options)

    def distill(self, options: DistillationOptions) -> TrainingRunResult:
        """Run knowledge distillation."""
        return self.dataset(options.dataset_name).distill(options)

    def domain_adapt(self, options: DomainAdaptationOptions) -> TrainingRunResult:
        """Run domain adaptation."""
        return self.dataset(options.dataset_name).domain_adapt(options)

    def grpo_train(self, options: GrpoOptions) -> TrainingRunResult:
        """Run GRPO training."""
        options = self._resolve_data_path(options, "grpo_data_path")
        if not options.dataset_name:
            return run_grpo_training(
                records=[], options=options, random_seed=42,
                data_root=self._config.data_root,
            )
        return self.dataset(options.dataset_name).grpo_train(options)

    def qlora_train(self, options: QloraOptions) -> TrainingRunResult:
        """Run QLoRA training."""
        options = self._resolve_data_path(options, "qlora_data_path")
        if not options.dataset_name:
            return run_qlora_training(
                records=[], options=options, random_seed=42,
                data_root=self._config.data_root,
            )
        return self.dataset(options.dataset_name).qlora_train(options)

    def kto_train(self, options: KtoOptions) -> TrainingRunResult:
        """Run KTO training."""
        options = self._resolve_data_path(options, "kto_data_path")
        if not options.dataset_name:
            return run_kto_training(
                records=[], options=options, random_seed=42,
                data_root=self._config.data_root,
            )
        return self.dataset(options.dataset_name).kto_train(options)

    def orpo_train(self, options: OrpoOptions) -> TrainingRunResult:
        """Run ORPO training."""
        options = self._resolve_data_path(options, "orpo_data_path")
        if not options.dataset_name:
            return run_orpo_training(
                records=[], options=options, random_seed=42,
                data_root=self._config.data_root,
            )
        return self.dataset(options.dataset_name).orpo_train(options)

    def multimodal_train(self, options: MultimodalOptions) -> TrainingRunResult:
        """Run multimodal fine-tuning."""
        options = self._resolve_data_path(options, "multimodal_data_path")
        if not options.dataset_name:
            return run_multimodal_training(
                records=[], options=options, random_seed=42,
                data_root=self._config.data_root,
            )
        return self.dataset(options.dataset_name).multimodal_train(options)

    def rlvr_train(self, options: RlvrOptions) -> TrainingRunResult:
        """Run RLVR training."""
        options = self._resolve_data_path(options, "rlvr_data_path")
        if not options.dataset_name:
            return run_rlvr_training(
                records=[], options=options, random_seed=42,
                data_root=self._config.data_root,
            )
        return self.dataset(options.dataset_name).rlvr_train(options)

    def chat(self, options: ChatOptions) -> ChatResult:
        """Run chat inference against a trained model."""
        if options.dataset_name is None:
            return run_chat(None, options)
        dataset = self.dataset(options.dataset_name)
        return dataset.chat(options)

    def with_data_root(self, data_root: str) -> "CrucibleClient":
        """Clone the client with a different local data root."""
        resolved_root = Path(data_root).expanduser().resolve()
        return CrucibleClient(replace(self._config, data_root=resolved_root))

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
    """SDK dataset handle."""

    def __init__(self, dataset_name: str, store: DatasetStore) -> None:
        self._dataset_name = dataset_name
        self._store = store

    @property
    def name(self) -> str:
        """Return dataset identifier."""
        return self._dataset_name

    def load_records(self) -> tuple[DatasetManifest, list[DataRecord]]:
        """Load dataset records."""
        return self._store.load_records(self._dataset_name)

    def export_training(
        self, output_dir: str,
        shard_size: int = 1000, include_metadata: bool = False,
    ) -> str:
        """Export dataset into local sharded training files."""
        request = TrainingExportRequest(
            dataset_name=self._dataset_name,
            output_dir=output_dir,
            shard_size=shard_size,
            include_metadata=include_metadata,
        )
        manifest_path = self._store.export_training_data(request)
        return str(manifest_path)

    def train(self, options: TrainingOptions) -> TrainingRunResult:
        """Train a model on this dataset with default/custom loop."""
        if options.dataset_name != self._dataset_name:
            raise CrucibleServeError(
                f"Training options dataset '{options.dataset_name}' does not match handle "
                f"'{self._dataset_name}'."
            )
        _, records = self.load_records()
        return run_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root,
        )

    def sft_train(self, options: SftOptions) -> TrainingRunResult:
        """Run supervised fine-tuning on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise CrucibleSftError(
                f"SFT dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        _, records = self.load_records()
        return run_sft_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root,
        )

    def lora_train(self, options: LoraTrainingOptions) -> TrainingRunResult:
        """Run LoRA fine-tuning on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise CrucibleLoraError(
                f"LoRA dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        return run_lora_training(
            options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root,
        )

    def dpo_train(self, options: DpoOptions) -> TrainingRunResult:
        """Run DPO preference optimization on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise CrucibleDpoError(
                f"DPO dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        _, records = self.load_records()
        return run_dpo_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root,
        )

    def rlhf_train(self, options: RlhfOptions) -> TrainingRunResult:
        """Run RLHF training with PPO on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise CrucibleRlhfError(
                f"RLHF dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        _, records = self.load_records()
        return run_rlhf_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root,
        )

    def distill(self, options: DistillationOptions) -> TrainingRunResult:
        """Run knowledge distillation on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise CrucibleDistillationError(
                f"Distillation dataset '{options.dataset_name}' != "
                f"handle '{self._dataset_name}'."
            )
        _, records = self.load_records()
        return run_distillation(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root,
        )

    def domain_adapt(self, options: DomainAdaptationOptions) -> TrainingRunResult:
        """Run domain adaptation on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise CrucibleServeError(
                f"Domain adaptation dataset '{options.dataset_name}' != "
                f"handle '{self._dataset_name}'."
            )
        _, records = self.load_records()
        return run_domain_adaptation(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root,
        )

    def grpo_train(self, options: GrpoOptions) -> TrainingRunResult:
        """Run GRPO training on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise CrucibleGrpoError(
                f"GRPO dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        _, records = self.load_records()
        return run_grpo_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root,
        )

    def qlora_train(self, options: QloraOptions) -> TrainingRunResult:
        """Run QLoRA training on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise CrucibleQloraError(
                f"QLoRA dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        _, records = self.load_records()
        return run_qlora_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root,
        )

    def kto_train(self, options: KtoOptions) -> TrainingRunResult:
        """Run KTO training on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise CrucibleKtoError(
                f"KTO dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        _, records = self.load_records()
        return run_kto_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root,
        )

    def orpo_train(self, options: OrpoOptions) -> TrainingRunResult:
        """Run ORPO training on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise CrucibleOrpoError(
                f"ORPO dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        _, records = self.load_records()
        return run_orpo_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root,
        )

    def multimodal_train(self, options: MultimodalOptions) -> TrainingRunResult:
        """Run multimodal fine-tuning on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise CrucibleMultimodalError(
                f"Multimodal dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        _, records = self.load_records()
        return run_multimodal_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root,
        )

    def rlvr_train(self, options: RlvrOptions) -> TrainingRunResult:
        """Run RLVR training on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise CrucibleRlvrError(
                f"RLVR dataset '{options.dataset_name}' != handle '{self._dataset_name}'."
            )
        _, records = self.load_records()
        return run_rlvr_training(
            records=records, options=options, random_seed=self._store.random_seed,
            data_root=self._store.data_root,
        )

    def chat(self, options: ChatOptions) -> ChatResult:
        """Run one chat completion on this dataset."""
        if options.dataset_name != self._dataset_name:
            raise CrucibleServeError(
                f"Chat options dataset '{options.dataset_name}' does not match handle "
                f"'{self._dataset_name}'."
            )
        _, records = self.load_records()
        return run_chat(records, options)
