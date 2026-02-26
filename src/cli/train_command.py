"""Train command wiring for Forge CLI.

This module isolates train command parser and execution logic.
It keeps the top-level CLI module focused and within size constraints.
"""

from __future__ import annotations

import argparse
from typing import cast

from core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_MAX_TOKEN_LENGTH,
    DEFAULT_POSITION_EMBEDDING_TYPE,
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
    DEFAULT_TRAIN_OPTIMIZER_TYPE,
    DEFAULT_TRAIN_PRECISION_MODE,
    DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS,
    DEFAULT_TRAIN_SCHEDULER_ETA_MIN,
    DEFAULT_TRAIN_SCHEDULER_GAMMA,
    DEFAULT_TRAIN_SCHEDULER_STEP_SIZE,
    DEFAULT_TRAIN_SCHEDULER_T_MAX_EPOCHS,
    DEFAULT_TRAIN_SCHEDULER_TYPE,
    DEFAULT_TRAIN_SGD_MOMENTUM,
    DEFAULT_TRAIN_VALIDATION_SPLIT,
    DEFAULT_TRAIN_WEIGHT_DECAY,
    SUPPORTED_POSITION_EMBEDDING_TYPES,
    SUPPORTED_TRAIN_OPTIMIZER_TYPES,
    SUPPORTED_TRAIN_PRECISION_MODES,
    SUPPORTED_TRAIN_SCHEDULER_TYPES,
)
from core.training_types import TrainingRunResult
from core.types import (
    OptimizerType,
    PositionEmbeddingType,
    PrecisionMode,
    SchedulerType,
    TrainingOptions,
)
from serve.hardware_auto_config import apply_hardware_auto_config
from store.dataset_sdk import ForgeClient


def run_train_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle train command invocation."""
    options = _build_training_options(args)
    if args.auto_config:
        options = apply_hardware_auto_config(options)
    result = client.train(options)
    _print_training_result(result)
    return 0


def _print_training_result(result: TrainingRunResult) -> None:
    """Print training result summary to stdout."""
    print(f"model_path={result.model_path}")
    print(f"history_path={result.history_path}")
    print(f"plot_path={result.plot_path or '-'}")
    print(f"epochs_completed={result.epochs_completed}")
    print(f"checkpoint_dir={result.checkpoint_dir or '-'}")
    print(f"best_checkpoint_path={result.best_checkpoint_path or '-'}")
    print(f"resumed_from_checkpoint={result.resumed_from_checkpoint or '-'}")
    print(f"run_id={result.run_id or '-'}")
    print(f"artifact_contract_path={result.artifact_contract_path or '-'}")


def _build_training_options(args: argparse.Namespace) -> TrainingOptions:
    """Build TrainingOptions from parsed CLI arguments."""
    return TrainingOptions(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        version_id=args.version_id,
        architecture_path=args.architecture_file,
        custom_loop_path=args.custom_loop_file,
        hooks_path=args.hooks_file,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        precision_mode=cast(PrecisionMode, args.precision_mode),
        optimizer_type=cast(OptimizerType, args.optimizer_type),
        weight_decay=args.weight_decay,
        sgd_momentum=args.sgd_momentum,
        scheduler_type=cast(SchedulerType, args.scheduler_type),
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        scheduler_t_max_epochs=args.scheduler_t_max_epochs,
        scheduler_eta_min=args.scheduler_eta_min,
        batch_size=args.batch_size,
        max_token_length=args.max_token_length,
        validation_split=args.validation_split,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        attention_heads=args.attention_heads,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_layers=args.mlp_layers,
        dropout=args.dropout,
        position_embedding_type=cast(
            PositionEmbeddingType,
            args.position_embedding_type,
        ),
        vocabulary_size=args.vocabulary_size,
        initial_weights_path=args.initial_weights_path,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        save_best_checkpoint=args.save_best_checkpoint,
        max_checkpoint_files=args.max_checkpoint_files,
        resume_checkpoint_path=args.resume_checkpoint_path,
        progress_log_interval_steps=args.progress_log_interval_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        auto_micro_batch=args.auto_micro_batch,
        gradient_checkpointing=args.gradient_checkpointing,
        tokenizer_path=args.tokenizer_path,
    )


def add_train_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register train subcommand."""
    parser = subparsers.add_parser(
        "train", help="Train a PyTorch language model on a dataset version"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--output-dir", required=True, help="Training artifact output directory")
    parser.add_argument("--version-id", help="Optional specific version id")
    parser.add_argument("--architecture-file", help="Optional .py or .json model architecture file")
    parser.add_argument("--custom-loop-file", help="Optional .py custom training loop file")
    parser.add_argument(
        "--hooks-file",
        help="Optional .py hook module with run/epoch/batch callback functions",
    )
    parser.add_argument(
        "--initial-weights-path",
        help="Optional model artifact (.pt or .onnx) used as initial weights",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS, help="Training epochs")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_TRAIN_LEARNING_RATE,
        help="Optimizer learning rate",
    )
    parser.add_argument("--precision-mode", default=DEFAULT_TRAIN_PRECISION_MODE,
                        choices=SUPPORTED_TRAIN_PRECISION_MODES, help="Mixed precision mode")
    parser.add_argument("--optimizer-type", default=DEFAULT_TRAIN_OPTIMIZER_TYPE,
                        choices=SUPPORTED_TRAIN_OPTIMIZER_TYPES, help="Optimizer backend")
    parser.add_argument("--weight-decay", type=float,
                        default=DEFAULT_TRAIN_WEIGHT_DECAY, help="Weight decay coefficient")
    parser.add_argument("--sgd-momentum", type=float,
                        default=DEFAULT_TRAIN_SGD_MOMENTUM, help="SGD momentum")
    parser.add_argument("--scheduler-type", default=DEFAULT_TRAIN_SCHEDULER_TYPE,
                        choices=SUPPORTED_TRAIN_SCHEDULER_TYPES, help="LR scheduler strategy")
    parser.add_argument("--scheduler-step-size", type=int,
                        default=DEFAULT_TRAIN_SCHEDULER_STEP_SIZE, help="Step scheduler interval")
    parser.add_argument("--scheduler-gamma", type=float,
                        default=DEFAULT_TRAIN_SCHEDULER_GAMMA, help="Step scheduler factor")
    parser.add_argument("--scheduler-t-max-epochs", type=int,
                        default=DEFAULT_TRAIN_SCHEDULER_T_MAX_EPOCHS, help="Cosine scheduler cycle")
    parser.add_argument("--scheduler-eta-min", type=float,
                        default=DEFAULT_TRAIN_SCHEDULER_ETA_MIN, help="Cosine scheduler min LR")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=DEFAULT_MAX_TOKEN_LENGTH,
        help="Maximum token length per record",
    )
    parser.add_argument(
        "--vocabulary-size", type=int, help="Optional maximum tokenizer vocabulary size"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_TRAIN_VALIDATION_SPLIT,
        help="Validation data fraction in [0,1)",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=DEFAULT_TRAIN_HIDDEN_DIM, help="Default model hidden size"
    )
    parser.add_argument(
        "--num-layers", type=int, default=DEFAULT_TRAIN_NUM_LAYERS, help="Default model layer count"
    )
    parser.add_argument(
        "--attention-heads",
        type=int,
        default=DEFAULT_TRAIN_ATTENTION_HEADS,
        help="Attention heads per transformer layer",
    )
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=DEFAULT_TRAIN_MLP_HIDDEN_DIM,
        help="Hidden width of transformer feed-forward block",
    )
    parser.add_argument(
        "--mlp-layers",
        type=int,
        default=DEFAULT_TRAIN_MLP_LAYERS,
        help="Number of MLP layers before vocabulary projection",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_TRAIN_DROPOUT,
        help="Dropout probability in default model",
    )
    parser.add_argument(
        "--position-embedding-type",
        default=DEFAULT_POSITION_EMBEDDING_TYPE,
        choices=SUPPORTED_POSITION_EMBEDDING_TYPES,
        help="Positional embedding mode for default model",
    )
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS,
        help="Save training checkpoint every N epochs",
    )
    parser.add_argument(
        "--max-checkpoint-files",
        type=int,
        default=DEFAULT_TRAIN_MAX_CHECKPOINT_FILES,
        help="Keep at most N epoch checkpoint files",
    )
    parser.add_argument(
        "--no-save-best-checkpoint",
        action="store_false",
        dest="save_best_checkpoint",
        help="Disable writing best.pt checkpoint",
    )
    parser.set_defaults(save_best_checkpoint=True)
    parser.add_argument(
        "--resume-checkpoint-path",
        help="Resume training state from a previously saved checkpoint file",
    )
    parser.add_argument(
        "--progress-log-interval-steps",
        type=int,
        default=DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS,
        help="Log training batch progress every N batches",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        help="Number of micro-batches to accumulate before optimizer step",
    )
    parser.add_argument(
        "--auto-micro-batch",
        action="store_true",
        default=False,
        help="Automatically probe GPU memory and set micro-batch size",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing for memory reduction",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Path to tokenizer or HuggingFace tokenizer ID",
    )
    parser.add_argument(
        "--auto-config",
        action="store_true",
        default=False,
        help="Auto-configure training defaults from detected hardware profile",
    )
