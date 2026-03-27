"""LoRA command wiring for Crucible CLI.

This module isolates LoRA training and merge command parser logic,
mapping CLI arguments to LoraTrainingOptions for adapter fine-tuning.
"""

from __future__ import annotations

import argparse

from cli.training_output import print_and_register
from core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    DEFAULT_MAX_TOKEN_LENGTH,
    DEFAULT_TRAIN_ATTENTION_HEADS,
    DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS,
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_TRAIN_HIDDEN_DIM,
    DEFAULT_TRAIN_LEARNING_RATE,
    DEFAULT_TRAIN_MLP_HIDDEN_DIM,
    DEFAULT_TRAIN_MLP_LAYERS,
    DEFAULT_TRAIN_NUM_LAYERS,
    DEFAULT_TRAIN_OPTIMIZER_TYPE,
    DEFAULT_TRAIN_PRECISION_MODE,
    DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS,
    DEFAULT_TRAIN_VALIDATION_SPLIT,
    DEFAULT_TRAIN_WEIGHT_DECAY,
    SUPPORTED_TRAIN_OPTIMIZER_TYPES,
    SUPPORTED_TRAIN_PRECISION_MODES,
)
from core.lora_types import LoraConfig, LoraTrainingOptions
from core.training_types import OptimizerType, PrecisionMode
from store.dataset_sdk import CrucibleClient


def run_lora_train_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Handle lora-train command invocation."""
    lora_config = LoraConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=tuple(args.lora_target_modules.split(",")),
    )
    options = LoraTrainingOptions(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        lora_data_path=args.lora_data_path,
        base_model_path=args.base_model_path,
        lora_config=lora_config,
        tokenizer_path=getattr(args, "tokenizer_path", None),
        resume_checkpoint_path=getattr(args, "resume_checkpoint_path", None),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_token_length=args.max_token_length,
        validation_split=args.validation_split,
        precision_mode=args.precision_mode,
        optimizer_type=args.optimizer_type,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        attention_heads=args.attention_heads,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_layers=args.mlp_layers,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        save_best_checkpoint=args.save_best_checkpoint,
        progress_log_interval_steps=args.progress_log_interval_steps,
        hooks_path=args.hooks_file,
    )
    result = client.lora_train(options)
    print_and_register(client, result, args.model_name)
    return 0


def run_lora_merge_command(args: argparse.Namespace) -> int:
    """Handle lora-merge command invocation."""
    from serve.lora_adapter_io import merge_lora_into_base
    from serve.lora_injection import inject_lora_adapters

    try:
        import torch
    except ImportError:
        print("Error: torch is required for lora-merge.")
        return 1

    from core.lora_types import LoraConfig

    config = LoraConfig(
        rank=args.lora_rank,
        target_modules=tuple(args.lora_target_modules.split(",")),
    )
    model_state = torch.load(args.base_model_path, map_location="cpu")
    print(f"Merge not yet fully implemented. Config: rank={config.rank}")
    return 0


def add_lora_train_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register lora-train subcommand."""
    parser = subparsers.add_parser(
        "lora-train", help="Fine-tune a model using LoRA adapters"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset", default="", help="Dataset name (auto-resolves data path)")
    parser.add_argument("--lora-data-path", default="", help="Path to JSONL file with training data")
    parser.add_argument("--base-model-path", required=True, help="Base model weights")
    parser.add_argument("--tokenizer-path", default=None, help="Path to tokenizer file (auto-detected if omitted)")
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=DEFAULT_LORA_ALPHA, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT, help="Dropout")
    parser.add_argument(
        "--lora-target-modules",
        default=",".join(("q_proj", "v_proj")),
        help="Comma-separated target module names",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_TRAIN_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-token-length", type=int, default=DEFAULT_MAX_TOKEN_LENGTH)
    parser.add_argument("--validation-split", type=float, default=DEFAULT_TRAIN_VALIDATION_SPLIT)
    parser.add_argument("--precision-mode", default=DEFAULT_TRAIN_PRECISION_MODE,
                        choices=SUPPORTED_TRAIN_PRECISION_MODES)
    parser.add_argument("--optimizer-type", default=DEFAULT_TRAIN_OPTIMIZER_TYPE,
                        choices=SUPPORTED_TRAIN_OPTIMIZER_TYPES)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_TRAIN_WEIGHT_DECAY)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_TRAIN_HIDDEN_DIM)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_TRAIN_NUM_LAYERS)
    parser.add_argument("--attention-heads", type=int, default=DEFAULT_TRAIN_ATTENTION_HEADS)
    parser.add_argument("--mlp-hidden-dim", type=int, default=DEFAULT_TRAIN_MLP_HIDDEN_DIM)
    parser.add_argument("--mlp-layers", type=int, default=DEFAULT_TRAIN_MLP_LAYERS)
    parser.add_argument("--checkpoint-every-epochs", type=int,
                        default=DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS)
    parser.add_argument("--no-save-best-checkpoint", action="store_false",
                        dest="save_best_checkpoint")
    parser.set_defaults(save_best_checkpoint=True)
    parser.add_argument("--progress-log-interval-steps", type=int,
                        default=DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS)
    parser.add_argument("--hooks-file", help="Optional hooks module path")
    parser.add_argument("--resume-checkpoint-path", default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--model-name", default=None, help="Name for model registry (auto-derived if not set)")
    parser.add_argument("--wandb-project", default=None, help="W&B project name for experiment tracking")
    parser.add_argument("--tensorboard-dir", default=None, help="TensorBoard log directory for experiment tracking")


def add_lora_merge_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register lora-merge subcommand."""
    parser = subparsers.add_parser(
        "lora-merge", help="Merge LoRA adapter into base model weights"
    )
    parser.add_argument("--base-model-path", required=True, help="Base model weights")
    parser.add_argument("--adapter-path", required=True, help="LoRA adapter weights")
    parser.add_argument("--output-path", required=True, help="Merged model output path")
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
