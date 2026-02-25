"""Chat command wiring for Forge CLI.

This module isolates chat command parser and execution logic.
It supports quick inference checks against trained model weights.
"""

from __future__ import annotations

import argparse
from typing import Any, cast

from core.chat_types import ChatOptions
from core.constants import (
    DEFAULT_CHAT_MAX_NEW_TOKENS,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_CHAT_TOP_K,
    DEFAULT_MAX_TOKEN_LENGTH,
    DEFAULT_POSITION_EMBEDDING_TYPE,
    DEFAULT_TRAIN_ATTENTION_HEADS,
    DEFAULT_TRAIN_DROPOUT,
    DEFAULT_TRAIN_HIDDEN_DIM,
    DEFAULT_TRAIN_MLP_HIDDEN_DIM,
    DEFAULT_TRAIN_MLP_LAYERS,
    DEFAULT_TRAIN_NUM_LAYERS,
    SUPPORTED_POSITION_EMBEDDING_TYPES,
)
from core.types import PositionEmbeddingType
from store.dataset_sdk import ForgeClient


def run_chat_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle chat command invocation."""
    options = ChatOptions(
        model_path=args.model_path,
        prompt=args.prompt,
        dataset_name=args.dataset,
        tokenizer_path=args.tokenizer_path,
        version_id=args.version_id,
        architecture_path=args.architecture_file,
        weights_path=getattr(args, "weights_path", None),
        max_new_tokens=args.max_new_tokens,
        max_token_length=args.max_token_length,
        temperature=args.temperature,
        top_k=args.top_k,
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
        stream=True,
    )
    client.chat(options)
    print()
    return 0


def add_chat_command(subparsers: Any) -> None:
    """Register chat subcommand."""
    parser = subparsers.add_parser(
        "chat",
        help="Generate a model response from trained weights and a text prompt",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name for tokenizer fallback (optional if vocab.json exists beside model)",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Path to tokenizer vocabulary JSON file (overrides --dataset and auto-detected vocab)",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="HuggingFace model ID (e.g. 'gpt2') or path to trained model file (.pt or .onnx)",
    )
    parser.add_argument(
        "--weights-path",
        default=None,
        help="Optional path to custom weights (.pt, .safetensors) to load into the model architecture",
    )
    parser.add_argument("--prompt", required=True, help="Prompt text to complete")
    parser.add_argument("--version-id", help="Optional specific dataset version id")
    parser.add_argument("--architecture-file", help="Optional .py or .json model architecture file")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_CHAT_MAX_NEW_TOKENS,
        help="Maximum number of generated tokens",
    )
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=DEFAULT_MAX_TOKEN_LENGTH,
        help="Maximum token context length",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_CHAT_TEMPERATURE,
        help="Sampling temperature; use 0 for greedy decoding",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_CHAT_TOP_K,
        help="Top-k sampling cutoff; set 0 to sample from full vocabulary",
    )
    parser.add_argument(
        "--vocabulary-size", type=int, help="Optional maximum tokenizer vocabulary size"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_TRAIN_HIDDEN_DIM,
        help="Default model hidden size",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_TRAIN_NUM_LAYERS,
        help="Default model layer count",
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
