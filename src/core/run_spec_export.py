"""Run-spec YAML export for training configuration.

This module serializes training options back into valid YAML run-spec
format, enabling round-trip export of completed training configurations.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from core.errors import CrucibleDependencyError
from core.types import TrainingOptions


def export_training_options_to_yaml(options: TrainingOptions) -> str:
    """Convert training options to a valid YAML run-spec string.

    Builds a version-1 run-spec with a single train step containing
    all non-default training parameters.

    Args:
        options: Training options to export.

    Returns:
        Valid YAML string representing the run-spec.

    Raises:
        CrucibleDependencyError: If PyYAML is unavailable.
    """
    yaml_module = _import_yaml()
    spec = _build_run_spec_dict(options)
    return yaml_module.dump(spec, default_flow_style=False, sort_keys=False)


def export_run_to_yaml(
    options: TrainingOptions,
    run_id: str | None = None,
) -> str:
    """Export a training run as a YAML run-spec with optional metadata.

    Args:
        options: Training options from the completed run.
        run_id: Optional run identifier to include as a comment.

    Returns:
        YAML string with run-spec and optional header comment.
    """
    yaml_body = export_training_options_to_yaml(options)
    if run_id:
        header = f"# Exported from training run: {run_id}\n"
        return header + yaml_body
    return yaml_body


def _build_run_spec_dict(options: TrainingOptions) -> dict[str, Any]:
    """Build run-spec dictionary from training options."""
    train_args = _training_options_to_step_args(options)
    return {
        "version": 1,
        "defaults": {
            "dataset": options.dataset_name,
        },
        "steps": [
            {
                "command": "train",
                **train_args,
            },
        ],
    }


def _training_options_to_step_args(options: TrainingOptions) -> dict[str, Any]:
    """Convert training options to step argument mapping.

    Includes all fields that differ from constructor defaults so the
    exported spec is self-contained and reproducible.
    """
    options_dict = asdict(options)
    args: dict[str, Any] = {}
    args["output_dir"] = options_dict["output_dir"]
    _copy_if_set(args, options_dict, "architecture_path", key_alias="architecture_file")
    _copy_if_set(args, options_dict, "custom_loop_path", key_alias="custom_loop_file")
    _copy_if_set(args, options_dict, "hooks_path", key_alias="hooks_file")
    _copy_if_set(args, options_dict, "initial_weights_path")
    _copy_if_set(args, options_dict, "resume_checkpoint_path")
    numeric_keys = [
        "epochs", "learning_rate", "batch_size", "max_token_length",
        "validation_split", "hidden_dim", "num_layers", "attention_heads",
        "mlp_hidden_dim", "mlp_layers", "dropout", "weight_decay",
        "sgd_momentum", "scheduler_step_size", "scheduler_gamma",
        "scheduler_eta_min", "checkpoint_every_epochs", "max_checkpoint_files",
        "progress_log_interval_steps", "gradient_accumulation_steps",
    ]
    for key in numeric_keys:
        if key in options_dict:
            args[key] = options_dict[key]
    string_keys = [
        "precision_mode", "optimizer_type", "scheduler_type",
        "position_embedding_type",
    ]
    for key in string_keys:
        if key in options_dict:
            args[key] = options_dict[key]
    _copy_if_set(args, options_dict, "vocabulary_size")
    _copy_if_set(args, options_dict, "scheduler_t_max_epochs")
    args["save_best_checkpoint"] = options_dict["save_best_checkpoint"]
    args["auto_micro_batch"] = options_dict["auto_micro_batch"]
    return args


def _copy_if_set(
    target: dict[str, Any],
    source: Mapping[str, Any],
    key: str,
    key_alias: str | None = None,
) -> None:
    """Copy a value from source to target if it is not None."""
    value = source.get(key)
    if value is not None:
        output_key = key_alias or key
        target[output_key] = value


def _import_yaml() -> Any:
    """Import PyYAML dependency."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as error:
        raise CrucibleDependencyError(
            "YAML export requires PyYAML. Install with pip install pyyaml."
        ) from error
    return yaml
