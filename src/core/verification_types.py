"""Typed models for verification workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from core.types import TrainingRunResult
from store.dataset_sdk import ForgeClient

VerificationMode = Literal["quick", "full"]
VerificationStatus = Literal["passed", "failed", "skipped"]


@dataclass(frozen=True)
class VerificationOptions:
    """Options controlling verification execution."""

    mode: VerificationMode
    source_path: str
    keep_artifacts: bool
    fail_fast: bool


@dataclass(frozen=True)
class VerificationCheckResult:
    """One verification check result row."""

    check_id: str
    title: str
    status: VerificationStatus
    details: str
    duration_seconds: float


@dataclass(frozen=True)
class VerificationReport:
    """Final verification report for a complete run."""

    mode: VerificationMode
    runtime_data_root: str
    artifacts_kept: bool
    checks: tuple[VerificationCheckResult, ...]

    @property
    def failed_count(self) -> int:
        """Count failed checks in this report."""
        return sum(1 for check in self.checks if check.status == "failed")

    @property
    def passed_count(self) -> int:
        """Count passed checks in this report."""
        return sum(1 for check in self.checks if check.status == "passed")


@dataclass
class VerificationRuntime:
    """Shared mutable runtime state used by check functions."""

    client: ForgeClient
    data_root: Path
    source_path: Path
    dataset_name: str
    run_spec_path: Path
    train_output_dir: Path
    export_output_dir: Path
    hooks_output_dir: Path
    hooks_marker_path: Path
    train_result: TrainingRunResult | None = None
    chat_preview: str | None = None
