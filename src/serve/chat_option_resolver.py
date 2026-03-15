"""Resolve chat-time model and tokenizer options.

This module keeps chat option translation isolated from generation logic.
It reconciles runtime flags, persisted training metadata, and checkpoint
shape inference so inference can reuse training-time configuration safely.
"""

from __future__ import annotations

from typing import Mapping, cast

from core.chat_types import ChatOptions, ChatTokenizer
from core.errors import CrucibleServeError
from core.types import DataRecord, PositionEmbeddingType, TrainingOptions
from serve.training_metadata import load_tokenizer, load_tokenizer_from_path, load_training_config
from serve.training_setup import fit_training_tokenizer


def resolve_chat_training_options(
    options: ChatOptions,
    model_state: Mapping[str, object],
) -> TrainingOptions:
    """Resolve training options used to construct chat-time model."""
    config_payload = load_training_config(options.model_path)
    if config_payload is not None:
        return _to_persisted_training_options(options, config_payload, model_state)
    if options.architecture_path is None:
        return _to_inferred_training_options(options, model_state)
    return _to_explicit_training_options(options)


def resolve_chat_tokenizer(
    records: list[DataRecord] | None,
    options: ChatOptions,
    training_options: TrainingOptions,
) -> ChatTokenizer:
    """Resolve tokenizer from explicit path, persisted artifacts, or dataset fallback.

    Priority order:
    1. Explicit --tokenizer-path provided by the user
    2. Persisted vocab.json located beside the model weights
    3. Rebuild from dataset records (requires records to be non-None)

    Raises:
        CrucibleServeError: If no tokenizer source is available.
    """
    if options.tokenizer_path is not None:
        return load_tokenizer_from_path(options.tokenizer_path)
    persisted_tokenizer = load_tokenizer(options.model_path)
    if persisted_tokenizer is not None:
        return persisted_tokenizer
    if records is None:
        raise CrucibleServeError(
            "No tokenizer found next to the model and no dataset provided. "
            "Provide --tokenizer-path pointing to a vocab.json file, "
            "or provide --dataset to rebuild the tokenizer from ingested data."
        )
    return fit_training_tokenizer(records, training_options)


def resolve_chat_model_vocab_size(
    tokenizer_vocabulary: Mapping[str, int],
    model_state: Mapping[str, object],
    training_options: TrainingOptions,
) -> int:
    """Resolve vocabulary size used for chat-time model construction."""
    tokenizer_size = len(tokenizer_vocabulary)
    inferred_vocab_size = _infer_shape_value(model_state, "embedding.weight", 0)
    if inferred_vocab_size is not None:
        return max(tokenizer_size, inferred_vocab_size)
    if training_options.vocabulary_size is not None:
        return max(tokenizer_size, training_options.vocabulary_size)
    return tokenizer_size


def _to_explicit_training_options(options: ChatOptions) -> TrainingOptions:
    """Build training options directly from chat overrides."""
    return TrainingOptions(
        dataset_name=options.dataset_name or "",
        output_dir=".",
        architecture_path=options.architecture_path,
        custom_loop_path=None,
        epochs=1,
        learning_rate=1e-3,
        batch_size=1,
        max_token_length=options.max_token_length,
        validation_split=0.1,
        hidden_dim=options.hidden_dim,
        num_layers=options.num_layers,
        attention_heads=options.attention_heads,
        mlp_hidden_dim=options.mlp_hidden_dim,
        mlp_layers=options.mlp_layers,
        dropout=options.dropout,
        position_embedding_type=options.position_embedding_type,
        vocabulary_size=options.vocabulary_size,
        initial_weights_path=options.model_path,
    )


def _to_persisted_training_options(
    options: ChatOptions,
    payload: Mapping[str, object],
    model_state: Mapping[str, object],
) -> TrainingOptions:
    """Build training options from persisted config with safe fallbacks.

    Cross-checks shape-sensitive values against the checkpoint state dict
    to guard against stale or incorrect config files.
    """
    max_token_length = _read_int(payload, "max_token_length", options.max_token_length)
    hidden_dim = (
        _infer_shape_value(model_state, "embedding.weight", 1)
        or _read_int(payload, "hidden_dim", options.hidden_dim)
    )
    num_layers = (
        _infer_encoder_layer_count(model_state)
        or _read_int(payload, "num_layers", options.num_layers)
    )
    mlp_hidden_dim = (
        _infer_shape_value(model_state, "encoder.layers.0.linear1.weight", 0)
        or _infer_shape_value(model_state, "blocks.0.ffn.0.weight", 0)
        or _read_int(payload, "mlp_hidden_dim", options.mlp_hidden_dim)
    )
    mlp_layers = (
        _infer_projection_layer_count(model_state)
        or _read_int(payload, "mlp_layers", options.mlp_layers)
    )
    dropout = _read_float(payload, "dropout", options.dropout)
    requested_heads = _read_int(payload, "attention_heads", options.attention_heads)
    attention_heads = _resolve_attention_heads(hidden_dim, requested_heads)
    vocabulary_size = _read_optional_int(payload, "vocabulary_size", options.vocabulary_size)
    inferred_position_type = _infer_position_embedding_type(
        model_state,
        options.position_embedding_type,
    )
    position_embedding_type = _read_position_embedding_type(
        payload,
        "position_embedding_type",
        inferred_position_type,
    )
    architecture_path = options.architecture_path or _read_optional_str(
        payload, "architecture_path"
    )
    return TrainingOptions(
        dataset_name=options.dataset_name or "",
        output_dir=".",
        architecture_path=architecture_path,
        custom_loop_path=None,
        epochs=1,
        learning_rate=1e-3,
        batch_size=1,
        max_token_length=max_token_length,
        validation_split=0.1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        attention_heads=attention_heads,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_layers=mlp_layers,
        dropout=dropout,
        position_embedding_type=position_embedding_type,
        vocabulary_size=vocabulary_size,
        initial_weights_path=options.model_path,
    )


def _to_inferred_training_options(
    options: ChatOptions,
    model_state: Mapping[str, object],
) -> TrainingOptions:
    """Build training options with shape inference from checkpoint state dict."""
    hidden_dim = _infer_shape_value(model_state, "embedding.weight", 1) or options.hidden_dim
    inferred_vocab_size = _infer_shape_value(model_state, "embedding.weight", 0)
    max_token_length = (
        _infer_shape_value(model_state, "position_embedding.weight", 0) or options.max_token_length
    )
    num_layers = _infer_encoder_layer_count(model_state) or options.num_layers
    mlp_hidden_dim = (
        _infer_shape_value(model_state, "encoder.layers.0.linear1.weight", 0)
        or _infer_shape_value(model_state, "blocks.0.ffn.0.weight", 0)
        or options.mlp_hidden_dim
    )
    mlp_layers = _infer_projection_layer_count(model_state) or options.mlp_layers
    attention_heads = _resolve_attention_heads(hidden_dim, options.attention_heads)
    vocabulary_size = options.vocabulary_size if options.vocabulary_size else inferred_vocab_size
    return TrainingOptions(
        dataset_name=options.dataset_name or "",
        output_dir=".",
        architecture_path=None,
        custom_loop_path=None,
        epochs=1,
        learning_rate=1e-3,
        batch_size=1,
        max_token_length=max_token_length,
        validation_split=0.1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        attention_heads=attention_heads,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_layers=mlp_layers,
        dropout=options.dropout,
        position_embedding_type=_infer_position_embedding_type(
            model_state, options.position_embedding_type
        ),
        vocabulary_size=vocabulary_size,
        initial_weights_path=options.model_path,
    )


def _infer_shape_value(model_state: Mapping[str, object], key: str, index: int) -> int | None:
    """Infer a tensor shape value from checkpoint state dict."""
    tensor_value = model_state.get(key)
    shape = getattr(tensor_value, "shape", None)
    if shape is None or len(shape) <= index:
        return None
    shape_value = int(shape[index])
    return shape_value if shape_value > 0 else None


def _infer_encoder_layer_count(model_state: Mapping[str, object]) -> int | None:
    """Infer transformer encoder layer count from checkpoint key prefixes.

    Supports both ``encoder.layers.N`` (PyTorch TransformerEncoder) and
    ``blocks.N`` (DefaultCausalModel) key formats.
    """
    layer_indexes: set[int] = set()
    for key in model_state:
        if key.startswith("encoder.layers."):
            parts = key.split(".")
            if len(parts) >= 3 and parts[2].isdigit():
                layer_indexes.add(int(parts[2]))
        elif key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                layer_indexes.add(int(parts[1]))
    if not layer_indexes:
        return None
    return max(layer_indexes) + 1


def _infer_projection_layer_count(model_state: Mapping[str, object]) -> int | None:
    """Infer output projection linear layer count from checkpoint keys."""
    if "output.weight" in model_state:
        return 1
    output_linear_layers = [
        key for key in model_state.keys() if key.startswith("output.") and key.endswith(".weight")
    ]
    return len(output_linear_layers) if output_linear_layers else None


def _resolve_attention_heads(hidden_dim: int, preferred_heads: int) -> int:
    """Resolve a valid attention head count for inferred hidden dimension."""
    if preferred_heads > 0 and hidden_dim % preferred_heads == 0:
        return preferred_heads
    for candidate in range(min(preferred_heads, hidden_dim), 0, -1):
        if hidden_dim % candidate == 0:
            return candidate
    return 1


def _infer_position_embedding_type(
    model_state: Mapping[str, object],
    fallback: PositionEmbeddingType,
) -> PositionEmbeddingType:
    """Infer positional embedding type from checkpoint keys."""
    if "position_embedding.weight" in model_state:
        return cast(PositionEmbeddingType, "learned")
    return fallback


def _read_int(payload: Mapping[str, object], field_name: str, default_value: int) -> int:
    """Read integer field from persisted payload."""
    raw_value = payload.get(field_name, default_value)
    return raw_value if isinstance(raw_value, int) else default_value


def _read_optional_int(
    payload: Mapping[str, object],
    field_name: str,
    default_value: int | None,
) -> int | None:
    """Read optional integer field from persisted payload."""
    raw_value = payload.get(field_name, default_value)
    if raw_value is None or isinstance(raw_value, int):
        return raw_value
    return default_value


def _read_float(payload: Mapping[str, object], field_name: str, default_value: float) -> float:
    """Read float field from persisted payload."""
    raw_value = payload.get(field_name, default_value)
    if isinstance(raw_value, (float, int)):
        return float(raw_value)
    return default_value


def _read_optional_str(
    payload: Mapping[str, object],
    field_name: str,
) -> str | None:
    """Read optional string field from persisted payload."""
    raw_value = payload.get(field_name)
    if raw_value is None or isinstance(raw_value, str):
        return raw_value
    return None


def _read_position_embedding_type(
    payload: Mapping[str, object],
    field_name: str,
    default_value: PositionEmbeddingType,
) -> PositionEmbeddingType:
    """Read position embedding type from persisted payload."""
    raw_value = payload.get(field_name, default_value)
    if isinstance(raw_value, str) and raw_value in {"learned", "sinusoidal"}:
        return cast(PositionEmbeddingType, raw_value)
    return default_value
