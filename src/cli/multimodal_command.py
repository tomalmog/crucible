"""Multimodal training command wiring."""

from __future__ import annotations

import argparse
from typing import cast

from cli.training_output import print_and_register
from core.constants import *
from core.multimodal_types import MultimodalOptions
from core.training_types import OptimizerType, PrecisionMode
from store.dataset_sdk import CrucibleClient


def run_multimodal_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    options = MultimodalOptions(
        dataset_name=args.dataset, output_dir=args.output_dir,
        multimodal_data_path=args.multimodal_data_path,
        image_encoder=args.image_encoder, image_size=args.image_size,
        projection_dim=args.projection_dim,
        epochs=args.epochs, learning_rate=args.learning_rate,
        batch_size=args.batch_size, max_token_length=args.max_token_length,
        validation_split=args.validation_split,
        precision_mode=cast(PrecisionMode, args.precision_mode),
        optimizer_type=cast(OptimizerType, args.optimizer_type),
        weight_decay=args.weight_decay, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, attention_heads=args.attention_heads,
        mlp_hidden_dim=args.mlp_hidden_dim, mlp_layers=args.mlp_layers,
        hooks_path=args.hooks_file, initial_weights_path=args.initial_weights_path,
        base_model=args.base_model,
        tokenizer_path=args.tokenizer_path,
        resume_checkpoint_path=args.resume_checkpoint_path,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        save_best_checkpoint=args.save_best_checkpoint,
        progress_log_interval_steps=args.progress_log_interval_steps,
    )
    result = client.multimodal_train(options)
    print_and_register(client, result, args.model_name)
    return 0


def add_multimodal_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("multimodal-train", help="Multimodal vision-language fine-tuning")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset", default="", help="Dataset name (auto-resolves data path)")
    parser.add_argument("--multimodal-data-path", default="", help="Path to image+text JSONL")
    parser.add_argument("--image-encoder", default="clip-vit-base", help="Vision encoder")
    parser.add_argument("--image-size", type=int, default=224, help="Image input size")
    parser.add_argument("--projection-dim", type=int, default=512, help="Projection dimension")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_TRAIN_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-token-length", type=int, default=DEFAULT_MAX_TOKEN_LENGTH)
    parser.add_argument("--validation-split", type=float, default=DEFAULT_TRAIN_VALIDATION_SPLIT)
    parser.add_argument("--precision-mode", default=DEFAULT_TRAIN_PRECISION_MODE, choices=SUPPORTED_TRAIN_PRECISION_MODES)
    parser.add_argument("--optimizer-type", default=DEFAULT_TRAIN_OPTIMIZER_TYPE, choices=SUPPORTED_TRAIN_OPTIMIZER_TYPES)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_TRAIN_WEIGHT_DECAY)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_TRAIN_HIDDEN_DIM)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_TRAIN_NUM_LAYERS)
    parser.add_argument("--attention-heads", type=int, default=DEFAULT_TRAIN_ATTENTION_HEADS)
    parser.add_argument("--mlp-hidden-dim", type=int, default=DEFAULT_TRAIN_MLP_HIDDEN_DIM)
    parser.add_argument("--mlp-layers", type=int, default=DEFAULT_TRAIN_MLP_LAYERS)
    parser.add_argument("--hooks-file", help="Hook module")
    parser.add_argument("--initial-weights-path", help="Initial weights")
    parser.add_argument("--base-model", help="HuggingFace model ID (e.g. 'gpt2') to use as base architecture")
    parser.add_argument("--checkpoint-every-epochs", type=int, default=DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS)
    parser.add_argument("--no-save-best-checkpoint", action="store_false", dest="save_best_checkpoint")
    parser.set_defaults(save_best_checkpoint=True)
    parser.add_argument("--progress-log-interval-steps", type=int, default=DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS)
    parser.add_argument("--tokenizer-path", default=None, help="Path to tokenizer or HuggingFace tokenizer ID")
    parser.add_argument("--resume-checkpoint-path", default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--model-name", default=None, help="Name for model registry (auto-derived if not set)")
    parser.add_argument("--wandb-project", default=None, help="W&B project name for experiment tracking")
    parser.add_argument("--tensorboard-dir", default=None, help="TensorBoard log directory for experiment tracking")
