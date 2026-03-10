"""Type-safe field parsing helpers for run-spec execution.

This module centralizes primitive parsing so run-spec executors can stay
concise and produce consistent validation errors across CLI and SDK flows.
"""

from __future__ import annotations

from typing import Mapping, cast

from core.constants import (
    DEFAULT_POSITION_EMBEDDING_TYPE,
    DEFAULT_TRAIN_OPTIMIZER_TYPE,
    DEFAULT_TRAIN_PRECISION_MODE,
    DEFAULT_TRAIN_SCHEDULER_TYPE,
    SUPPORTED_POSITION_EMBEDDING_TYPES,
    SUPPORTED_TRAIN_OPTIMIZER_TYPES,
    SUPPORTED_TRAIN_PRECISION_MODES,
    SUPPORTED_TRAIN_SCHEDULER_TYPES,
)
from core.errors import CrucibleRunSpecError
from core.types import OptimizerType, PositionEmbeddingType, PrecisionMode, SchedulerType


def required_string(args: Mapping[str, object], field_name: str) -> str:
    """Read a required string field from a run-spec step."""
    value = optional_string(args, field_name)
    if value is None:
        raise CrucibleRunSpecError(f"Run-spec step is missing required field '{field_name}'.")
    return value


def optional_string(args: Mapping[str, object], field_name: str) -> str | None:
    """Read an optional string field from a run-spec step."""
    value = args.get(field_name)
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    raise CrucibleRunSpecError(f"Run-spec field '{field_name}' must be a string when provided.")


def optional_int(args: Mapping[str, object], field_name: str) -> int | None:
    """Read an optional integer field from a run-spec step."""
    value = args.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool):
        raise CrucibleRunSpecError(f"Run-spec field '{field_name}' must be an integer.")
    if isinstance(value, int):
        return value
    raise CrucibleRunSpecError(f"Run-spec field '{field_name}' must be an integer.")


def int_with_default(args: Mapping[str, object], field_name: str, default_value: int) -> int:
    """Read an integer field while preserving explicit zero values."""
    value = optional_int(args, field_name)
    return default_value if value is None else value


def optional_float(args: Mapping[str, object], field_name: str) -> float | None:
    """Read an optional numeric field from a run-spec step."""
    value = args.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool):
        raise CrucibleRunSpecError(f"Run-spec field '{field_name}' must be numeric.")
    if isinstance(value, (int, float)):
        return float(value)
    raise CrucibleRunSpecError(f"Run-spec field '{field_name}' must be numeric.")


def float_with_default(args: Mapping[str, object], field_name: str, default_value: float) -> float:
    """Read a numeric field while preserving explicit zero values."""
    value = optional_float(args, field_name)
    return default_value if value is None else value


def optional_bool(
    args: Mapping[str, object],
    field_name: str,
    default_value: bool,
) -> bool:
    """Read an optional boolean field from a run-spec step."""
    value = args.get(field_name)
    if value is None:
        return default_value
    if isinstance(value, bool):
        return value
    raise CrucibleRunSpecError(f"Run-spec field '{field_name}' must be true/false.")


def parse_position_embedding_type(args: Mapping[str, object]) -> PositionEmbeddingType:
    """Parse optional positional embedding mode from step arguments."""
    value = optional_string(args, "position_embedding_type")
    if value is None:
        return cast(PositionEmbeddingType, DEFAULT_POSITION_EMBEDDING_TYPE)
    if value in SUPPORTED_POSITION_EMBEDDING_TYPES:
        return cast(PositionEmbeddingType, value)
    supported_rows = ", ".join(SUPPORTED_POSITION_EMBEDDING_TYPES)
    raise CrucibleRunSpecError(
        f"Invalid position_embedding_type '{value}'. Use one of: {supported_rows}."
    )


def parse_optimizer_type(args: Mapping[str, object]) -> OptimizerType:
    """Parse optional optimizer type from step arguments."""
    value = optional_string(args, "optimizer_type")
    if value is None:
        return cast(OptimizerType, DEFAULT_TRAIN_OPTIMIZER_TYPE)
    if value in SUPPORTED_TRAIN_OPTIMIZER_TYPES:
        return cast(OptimizerType, value)
    supported_rows = ", ".join(SUPPORTED_TRAIN_OPTIMIZER_TYPES)
    raise CrucibleRunSpecError(f"Invalid optimizer_type '{value}'. Use one of: {supported_rows}.")


def parse_precision_mode(args: Mapping[str, object]) -> PrecisionMode:
    """Parse optional precision mode from step arguments."""
    value = optional_string(args, "precision_mode")
    if value is None:
        return cast(PrecisionMode, DEFAULT_TRAIN_PRECISION_MODE)
    if value in SUPPORTED_TRAIN_PRECISION_MODES:
        return cast(PrecisionMode, value)
    supported_rows = ", ".join(SUPPORTED_TRAIN_PRECISION_MODES)
    raise CrucibleRunSpecError(f"Invalid precision_mode '{value}'. Use one of: {supported_rows}.")


def parse_scheduler_type(args: Mapping[str, object]) -> SchedulerType:
    """Parse optional scheduler type from step arguments."""
    value = optional_string(args, "scheduler_type")
    if value is None:
        return cast(SchedulerType, DEFAULT_TRAIN_SCHEDULER_TYPE)
    if value in SUPPORTED_TRAIN_SCHEDULER_TYPES:
        return cast(SchedulerType, value)
    supported_rows = ", ".join(SUPPORTED_TRAIN_SCHEDULER_TYPES)
    raise CrucibleRunSpecError(f"Invalid scheduler_type '{value}'. Use one of: {supported_rows}.")
