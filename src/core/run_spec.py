"""Typed run-spec parsing for declarative Crucible pipelines.

This module loads and validates YAML run-spec files used by CLI workflows.
It provides one strict schema so future UI, CLI, and SDK execution paths
can consume the same pipeline description safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence, cast

from core.errors import CrucibleDependencyError, CrucibleRunSpecError

RunSpecCommand = Literal[
    "ingest",
    "filter",
    "train",
    "sft-train",
    "dpo-train",
    "rlhf-train",
    "distill",
    "distributed-train",
    "domain-adapt",
    "lora-train",
    "lora-merge",
    "export-training",
    "chat",
    "versions",
    "hardware-profile",
    "benchmark",
    "sweep",
]
SUPPORTED_RUN_SPEC_COMMANDS: tuple[RunSpecCommand, ...] = (
    "ingest",
    "filter",
    "train",
    "sft-train",
    "dpo-train",
    "rlhf-train",
    "distill",
    "distributed-train",
    "domain-adapt",
    "lora-train",
    "lora-merge",
    "export-training",
    "chat",
    "versions",
    "hardware-profile",
    "benchmark",
    "sweep",
)


@dataclass(frozen=True)
class RunSpecDefaults:
    """Default values applied to run-spec steps."""

    data_root: str | None = None
    dataset_name: str | None = None


@dataclass(frozen=True)
class RunSpecStep:
    """One runnable pipeline step from a run-spec file."""

    command: RunSpecCommand
    args: Mapping[str, object]


@dataclass(frozen=True)
class RunSpec:
    """Validated run-spec root object."""

    version: int
    defaults: RunSpecDefaults
    steps: tuple[RunSpecStep, ...]


def load_run_spec(spec_path: str) -> RunSpec:
    """Load and validate a YAML run-spec from disk.

    Args:
        spec_path: File path to YAML run-spec.

    Returns:
        Fully validated run-spec object.

    Raises:
        CrucibleDependencyError: If PyYAML is unavailable.
        CrucibleRunSpecError: If file is invalid or schema checks fail.
    """
    payload = _load_yaml_payload(spec_path)
    root_mapping = _expect_mapping(payload, "run spec root")
    version = _parse_version(root_mapping)
    defaults = _parse_defaults(root_mapping)
    steps = _parse_steps(root_mapping)
    _validate_root_keys(root_mapping)
    return RunSpec(version=version, defaults=defaults, steps=steps)


def _load_yaml_payload(spec_path: str) -> object:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - dependency failure
        raise CrucibleDependencyError(
            "YAML run-spec support requires PyYAML. Install with 'pip install pyyaml==6.0.2'."
        ) from error
    spec_file = Path(spec_path).expanduser().resolve()
    if not spec_file.exists():
        raise CrucibleRunSpecError(
            f"Run spec file does not exist at {spec_file}. Provide a valid YAML file path."
        )
    try:
        payload = cast(object, yaml.safe_load(spec_file.read_text(encoding="utf-8")))
    except OSError as error:
        raise CrucibleRunSpecError(
            f"Failed to read run spec at {spec_file}: {error}. Check file permissions and retry."
        ) from error
    except Exception as error:
        raise CrucibleRunSpecError(
            f"Failed to parse YAML run spec at {spec_file}: {error}. Fix YAML syntax and retry."
        ) from error
    if payload is None:
        raise CrucibleRunSpecError(f"Run spec at {spec_file} is empty. Define 'version' and 'steps'.")
    return payload


def _expect_mapping(value: object, context: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        normalized_mapping = {}
        for key, payload in value.items():
            if not isinstance(key, str):
                raise CrucibleRunSpecError(
                    f"Invalid {context}: expected string keys, got {type(key).__name__}."
                )
            normalized_mapping[key] = payload
        return normalized_mapping
    raise CrucibleRunSpecError(
        f"Invalid {context}: expected object mapping, got {type(value).__name__}."
    )


def _expect_sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise CrucibleRunSpecError(f"Invalid {context}: expected list, got {type(value).__name__}.")


def _parse_version(root_mapping: Mapping[str, object]) -> int:
    raw_version = root_mapping.get("version")
    if not isinstance(raw_version, int):
        raise CrucibleRunSpecError("Run spec field 'version' must be an integer. Set version: 1.")
    if raw_version != 1:
        raise CrucibleRunSpecError(f"Unsupported run spec version {raw_version}. Use version: 1.")
    return raw_version


def _parse_defaults(root_mapping: Mapping[str, object]) -> RunSpecDefaults:
    raw_defaults = root_mapping.get("defaults")
    if raw_defaults is None:
        return RunSpecDefaults()
    defaults_mapping = _expect_mapping(raw_defaults, "run spec defaults")
    _validate_defaults_keys(defaults_mapping)
    data_root = _optional_string(defaults_mapping, "data_root")
    dataset_name = _optional_string(defaults_mapping, "dataset")
    return RunSpecDefaults(data_root=data_root, dataset_name=dataset_name)


def _parse_steps(root_mapping: Mapping[str, object]) -> tuple[RunSpecStep, ...]:
    raw_steps = root_mapping.get("steps")
    if raw_steps is None:
        raise CrucibleRunSpecError(
            "Run spec missing required field 'steps'. Add a non-empty list of commands."
        )
    step_rows = _expect_sequence(raw_steps, "run spec steps")
    if len(step_rows) == 0:
        raise CrucibleRunSpecError("Run spec field 'steps' must include at least one step.")
    parsed_steps = []
    for index, step_value in enumerate(step_rows):
        parsed_steps.append(_parse_step(step_value, index))
    return tuple(parsed_steps)


def _parse_step(step_value: object, step_index: int) -> RunSpecStep:
    context = f"run spec step #{step_index + 1}"
    step_mapping = _expect_mapping(step_value, context)
    raw_command = step_mapping.get("command")
    if not isinstance(raw_command, str):
        raise CrucibleRunSpecError(f"Invalid {context}: field 'command' must be a string.")
    command = _parse_command(raw_command, context)
    args_mapping = _parse_step_args(step_mapping, context)
    return RunSpecStep(command=command, args=args_mapping)


def _parse_command(raw_command: str, context: str) -> RunSpecCommand:
    if raw_command in SUPPORTED_RUN_SPEC_COMMANDS:
        return cast(RunSpecCommand, raw_command)
    supported_rows = ", ".join(SUPPORTED_RUN_SPEC_COMMANDS)
    raise CrucibleRunSpecError(
        f"Unsupported command '{raw_command}' in {context}. Use one of: {supported_rows}."
    )


def _parse_step_args(step_mapping: Mapping[str, object], context: str) -> Mapping[str, object]:
    if "args" in step_mapping:
        if len(step_mapping.keys() - {"command", "args"}) > 0:
            raise CrucibleRunSpecError(
                f"Invalid {context}: when using 'args', do not mix inline keys."
            )
        raw_args = step_mapping["args"]
        return _expect_mapping(raw_args, f"{context} args")
    inline_args = {}
    for key, value in step_mapping.items():
        if key == "command":
            continue
        inline_args[key] = value
    return inline_args


def _optional_string(mapping: Mapping[str, object], field_name: str) -> str | None:
    raw_value = mapping.get(field_name)
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        normalized_value = raw_value.strip()
        return normalized_value if normalized_value else None
    raise CrucibleRunSpecError(f"Run spec field '{field_name}' must be a string when provided.")


def _validate_root_keys(root_mapping: Mapping[str, object]) -> None:
    allowed_keys = {"version", "defaults", "steps"}
    unknown_keys = sorted(set(root_mapping) - allowed_keys)
    if unknown_keys:
        raise CrucibleRunSpecError(
            f"Run spec contains unknown root fields: {', '.join(unknown_keys)}."
        )


def _validate_defaults_keys(defaults_mapping: Mapping[str, object]) -> None:
    allowed_keys = {"data_root", "dataset"}
    unknown_keys = sorted(set(defaults_mapping) - allowed_keys)
    if unknown_keys:
        raise CrucibleRunSpecError(
            f"Run spec defaults contain unknown fields: {', '.join(unknown_keys)}."
        )
