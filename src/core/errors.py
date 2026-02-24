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
