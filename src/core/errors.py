"""Forge exception hierarchy.

This module defines traceable domain errors with clear boundaries.
Each subsystem raises a specific error type for debuggability.
"""

from __future__ import annotations


class ForgeError(Exception):
    """Base exception for all Forge failures."""


class ForgeConfigError(ForgeError):
    """Raised for invalid runtime configuration."""


class ForgeIngestError(ForgeError):
    """Raised for source parsing and ingest failures."""


class ForgeTransformError(ForgeError):
    """Raised for transform pipeline failures."""


class ForgeStoreError(ForgeError):
    """Raised for dataset store and versioning failures."""


class ForgeServeError(ForgeError):
    """Raised for training data serving failures."""


class ForgeTrainingDivergedError(ForgeServeError):
    """Raised when training loss becomes NaN or Inf."""


class ForgeDependencyError(ForgeError):
    """Raised when an optional runtime dependency is missing."""


class ForgeRunSpecError(ForgeError):
    """Raised for invalid or unsupported run-spec configuration."""


class ForgeVerificationError(ForgeError):
    """Raised when automated verification checks fail."""


class ForgeSftError(ForgeError):
    """Raised for SFT training failures."""


class ForgeDpoError(ForgeError):
    """Raised for DPO training failures."""


class ForgeLoraError(ForgeError):
    """Raised for LoRA adapter failures."""


class ForgeQloraError(ForgeError):
    """Raised for QLoRA quantized training failures."""


class ForgeRlhfError(ForgeError):
    """Raised for RLHF training failures."""


class ForgeDistillationError(ForgeError):
    """Raised for knowledge distillation failures."""


class ForgeDistributedError(ForgeError):
    """Raised for distributed training failures."""


class ForgeSweepError(ForgeError):
    """Raised for hyperparameter sweep failures."""


class ForgeBenchmarkError(ForgeError):
    """Raised for benchmark evaluation failures."""


class ForgeSafetyError(ForgeError):
    """Raised for safety evaluation failures."""


class ForgeDeployError(ForgeError):
    """Raised for deployment packaging failures."""


class ForgeComputeError(ForgeError):
    """Raised for compute connectivity failures."""


class ForgeServerError(ForgeError):
    """Raised for collaboration server failures."""


class ForgeModelRegistryError(ForgeError):
    """Raised for model registry failures."""


class ForgeGrpoError(ForgeError):
    """Raised for GRPO training failures."""


class ForgeKtoError(ForgeError):
    """Raised for KTO training failures."""


class ForgeOrpoError(ForgeError):
    """Raised for ORPO training failures."""


class ForgeMultimodalError(ForgeError):
    """Raised for multimodal training failures."""


class ForgeRlvrError(ForgeError):
    """Raised for RLVR training failures."""


class ForgeExperimentError(ForgeError):
    """Raised for experiment tracking failures."""


class ForgeHubError(ForgeError):
    """Raised for HuggingFace Hub operation failures."""


class ForgeEvalError(ForgeError):
    """Raised for evaluation harness failures."""


class ForgeCurateError(ForgeError):
    """Raised for dataset curation failures."""


class ForgeMergeError(ForgeError):
    """Raised for model merging failures."""


class ForgeCloudError(ForgeError):
    """Raised for cloud burst operation failures."""


class ForgeCostError(ForgeError):
    """Raised for cost tracking failures."""


class ForgeSyntheticError(ForgeError):
    """Raised for synthetic data generation failures."""


class ForgeRecipeError(ForgeError):
    """Raised for training recipe failures."""


class ForgeJudgeError(ForgeError):
    """Raised for LLM-as-judge evaluation failures."""


class ForgeRemoteError(ForgeError):
    """Raised for remote Slurm cluster operation failures."""
