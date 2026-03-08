"""Run-spec option object builders.

This module converts raw run-spec step arguments into typed option dataclasses.
It keeps parsing logic isolated from run-spec orchestration.
"""

from __future__ import annotations

from typing import Mapping

from core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_TOKEN_LENGTH,
    DEFAULT_TRAIN_ATTENTION_HEADS,
    DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS,
    DEFAULT_TRAIN_DROPOUT,
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_TRAIN_HIDDEN_DIM,
    DEFAULT_TRAIN_LEARNING_RATE,
    DEFAULT_TRAIN_MAX_CHECKPOINT_FILES,
    DEFAULT_TRAIN_MLP_HIDDEN_DIM,
    DEFAULT_TRAIN_MLP_LAYERS,
    DEFAULT_TRAIN_NUM_LAYERS,
    DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS,
    DEFAULT_TRAIN_SCHEDULER_ETA_MIN,
    DEFAULT_TRAIN_SCHEDULER_GAMMA,
    DEFAULT_TRAIN_SCHEDULER_STEP_SIZE,
    DEFAULT_TRAIN_SCHEDULER_T_MAX_EPOCHS,
    DEFAULT_TRAIN_SGD_MOMENTUM,
    DEFAULT_TRAIN_VALIDATION_SPLIT,
    DEFAULT_TRAIN_WEIGHT_DECAY,
)
from core.run_spec_fields import (
    float_with_default,
    int_with_default,
    optional_bool,
    optional_int,
    optional_string,
    parse_optimizer_type,
    parse_position_embedding_type,
    parse_precision_mode,
    parse_scheduler_type,
    required_string,
)
from core.types import TrainingOptions


def build_training_options_for_run_spec(
    args: Mapping[str, object],
    dataset_name: str,
) -> TrainingOptions:
    """Build training options from one run-spec train step."""
    scheduler_type = parse_scheduler_type(args)
    scheduler_t_max_epochs = optional_int(args, "scheduler_t_max_epochs")
    if scheduler_type == "none":
        scheduler_t_max_epochs = DEFAULT_TRAIN_SCHEDULER_T_MAX_EPOCHS
    return TrainingOptions(
        dataset_name=dataset_name,
        output_dir=required_string(args, "output_dir"),
        architecture_path=optional_string(args, "architecture_file"),
        custom_loop_path=optional_string(args, "custom_loop_file"),
        hooks_path=optional_string(args, "hooks_file"),
        epochs=int_with_default(args, "epochs", DEFAULT_TRAIN_EPOCHS),
        learning_rate=float_with_default(args, "learning_rate", DEFAULT_TRAIN_LEARNING_RATE),
        precision_mode=parse_precision_mode(args),
        optimizer_type=parse_optimizer_type(args),
        weight_decay=float_with_default(args, "weight_decay", DEFAULT_TRAIN_WEIGHT_DECAY),
        sgd_momentum=float_with_default(args, "sgd_momentum", DEFAULT_TRAIN_SGD_MOMENTUM),
        scheduler_type=scheduler_type,
        scheduler_step_size=int_with_default(
            args,
            "scheduler_step_size",
            DEFAULT_TRAIN_SCHEDULER_STEP_SIZE,
        ),
        scheduler_gamma=float_with_default(args, "scheduler_gamma", DEFAULT_TRAIN_SCHEDULER_GAMMA),
        scheduler_t_max_epochs=scheduler_t_max_epochs,
        scheduler_eta_min=float_with_default(
            args,
            "scheduler_eta_min",
            DEFAULT_TRAIN_SCHEDULER_ETA_MIN,
        ),
        batch_size=int_with_default(args, "batch_size", DEFAULT_BATCH_SIZE),
        max_token_length=int_with_default(args, "max_token_length", DEFAULT_MAX_TOKEN_LENGTH),
        validation_split=float_with_default(
            args,
            "validation_split",
            DEFAULT_TRAIN_VALIDATION_SPLIT,
        ),
        hidden_dim=int_with_default(args, "hidden_dim", DEFAULT_TRAIN_HIDDEN_DIM),
        num_layers=int_with_default(args, "num_layers", DEFAULT_TRAIN_NUM_LAYERS),
        attention_heads=int_with_default(args, "attention_heads", DEFAULT_TRAIN_ATTENTION_HEADS),
        mlp_hidden_dim=int_with_default(args, "mlp_hidden_dim", DEFAULT_TRAIN_MLP_HIDDEN_DIM),
        mlp_layers=int_with_default(args, "mlp_layers", DEFAULT_TRAIN_MLP_LAYERS),
        dropout=float_with_default(args, "dropout", DEFAULT_TRAIN_DROPOUT),
        position_embedding_type=parse_position_embedding_type(args),
        vocabulary_size=optional_int(args, "vocabulary_size"),
        initial_weights_path=optional_string(args, "initial_weights_path"),
        checkpoint_every_epochs=int_with_default(
            args,
            "checkpoint_every_epochs",
            DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS,
        ),
        save_best_checkpoint=optional_bool(args, "save_best_checkpoint", default_value=True),
        max_checkpoint_files=int_with_default(
            args,
            "max_checkpoint_files",
            DEFAULT_TRAIN_MAX_CHECKPOINT_FILES,
        ),
        resume_checkpoint_path=optional_string(args, "resume_checkpoint_path"),
        progress_log_interval_steps=int_with_default(
            args,
            "progress_log_interval_steps",
            DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS,
        ),
    )
