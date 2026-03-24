"""Crucible exception hierarchy.

This module defines traceable domain errors with clear boundaries.
Each subsystem raises a specific error type for debuggability.
"""

from __future__ import annotations


class CrucibleError(Exception):
    """Base exception for all Crucible failures."""


class CrucibleConfigError(CrucibleError):
    """Raised for invalid runtime configuration."""


class CrucibleIngestError(CrucibleError):
    """Raised for source parsing and ingest failures."""


class CrucibleTransformError(CrucibleError):
    """Raised for transform pipeline failures."""


class CrucibleStoreError(CrucibleError):
    """Raised for dataset store and versioning failures."""


class CrucibleServeError(CrucibleError):
    """Raised for training data serving failures."""


class CrucibleTrainingDivergedError(CrucibleServeError):
    """Raised when training loss becomes NaN or Inf."""


class CrucibleDependencyError(CrucibleError):
    """Raised when an optional runtime dependency is missing."""


class CrucibleRunSpecError(CrucibleError):
    """Raised for invalid or unsupported run-spec configuration."""


class CrucibleVerificationError(CrucibleError):
    """Raised when automated verification checks fail."""


class CrucibleSftError(CrucibleError):
    """Raised for SFT training failures."""


class CrucibleDpoError(CrucibleError):
    """Raised for DPO training failures."""


class CrucibleLoraError(CrucibleError):
    """Raised for LoRA adapter failures."""


class CrucibleQloraError(CrucibleError):
    """Raised for QLoRA quantized training failures."""


class CrucibleRlhfError(CrucibleError):
    """Raised for RLHF training failures."""


class CrucibleDistillationError(CrucibleError):
    """Raised for knowledge distillation failures."""


class CrucibleDistributedError(CrucibleError):
    """Raised for distributed training failures."""


class CrucibleSweepError(CrucibleError):
    """Raised for hyperparameter sweep failures."""


class CrucibleBenchmarkError(CrucibleError):
    """Raised for benchmark evaluation failures."""


class CrucibleComputeError(CrucibleError):
    """Raised for compute connectivity failures."""


class CrucibleServerError(CrucibleError):
    """Raised for collaboration server failures."""


class CrucibleModelRegistryError(CrucibleError):
    """Raised for model registry failures."""


class CrucibleGrpoError(CrucibleError):
    """Raised for GRPO training failures."""


class CrucibleKtoError(CrucibleError):
    """Raised for KTO training failures."""


class CrucibleOrpoError(CrucibleError):
    """Raised for ORPO training failures."""


class CrucibleMultimodalError(CrucibleError):
    """Raised for multimodal training failures."""


class CrucibleRlvrError(CrucibleError):
    """Raised for RLVR training failures."""


class CrucibleExperimentError(CrucibleError):
    """Raised for experiment tracking failures."""


class CrucibleHubError(CrucibleError):
    """Raised for HuggingFace Hub operation failures."""


class CrucibleEvalError(CrucibleError):
    """Raised for evaluation harness failures."""


class CrucibleCurateError(CrucibleError):
    """Raised for dataset curation failures."""


class CrucibleMergeError(CrucibleError):
    """Raised for model merging failures."""


class CrucibleCloudError(CrucibleError):
    """Raised for cloud burst operation failures."""


class CrucibleCostError(CrucibleError):
    """Raised for cost tracking failures."""


class CrucibleSyntheticError(CrucibleError):
    """Raised for synthetic data generation failures."""


class CrucibleRecipeError(CrucibleError):
    """Raised for training recipe failures."""


class CrucibleJudgeError(CrucibleError):
    """Raised for LLM-as-judge evaluation failures."""


class CrucibleRemoteError(CrucibleError):
    """Raised for remote Slurm cluster operation failures."""


class CrucibleExportError(CrucibleError):
    """Raised for model export failures."""
