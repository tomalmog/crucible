"""Training setup helpers for tokenizer and option validation.

This module centralizes pre-loop setup checks for training runs.
It keeps validation and tokenizer fitting isolated from loop execution.
"""

from __future__ import annotations

from pathlib import Path

from core.constants import (
    SUPPORTED_TRAIN_OPTIMIZER_TYPES,
    SUPPORTED_TRAIN_PRECISION_MODES,
    SUPPORTED_TRAIN_SCHEDULER_TYPES,
)
from core.chat_types import ChatTokenizer
from core.errors import CrucibleServeError
from core.types import DataRecord, TrainingOptions
from serve.tokenization import VocabularyTokenizer


def resolve_tokenizer(
    tokenizer_path: str | None = None,
    base_model: str | None = None,
) -> ChatTokenizer | None:
    """Resolve a pre-trained tokenizer from an explicit path or HF model ID.

    Returns a tokenizer satisfying the ChatTokenizer protocol if one can
    be loaded from the given path or base model, or None if neither is
    available.  All HuggingFace tokenizers are wrapped so that
    .encode(text, max_length) uses the Crucible signature.
    """
    if tokenizer_path:
        return _load_tokenizer_from_option(tokenizer_path)
    if base_model:
        from serve.hf_model_loader import is_huggingface_model_id

        if is_huggingface_model_id(base_model):
            return _load_hf_tokenizer_as_vocabulary(base_model)
    return None


def fit_training_tokenizer(
    records: list[DataRecord],
    options: TrainingOptions,
    base_model: str | None = None,
) -> ChatTokenizer:
    """Build or load tokenizer vocabulary for training.

    When ``options.tokenizer_path`` is set, loads a pre-trained tokenizer
    from the given path instead of fitting a new one from record texts.
    When ``base_model`` is a HuggingFace model ID and no explicit
    tokenizer path is provided, auto-loads the HF tokenizer (preserving
    BPE subword encoding).
    """
    resolved = resolve_tokenizer(options.tokenizer_path, base_model)
    if resolved is not None:
        return resolved
    tokenizer = VocabularyTokenizer.create()
    tokenizer.fit(
        (record.text for record in records),
        max_vocabulary_size=options.vocabulary_size,
    )
    return tokenizer


def _load_tokenizer_from_option(tokenizer_path: str) -> ChatTokenizer:
    """Load a pre-trained tokenizer from an explicit path.

    Supports Crucible vocab.json files and HuggingFace tokenizer.json files.
    Returns a tokenizer satisfying the ChatTokenizer protocol, preserving
    the original encoding (BPE for HuggingFace, whitespace for Crucible).
    """
    from serve.training_metadata import load_tokenizer_from_path

    loaded = load_tokenizer_from_path(tokenizer_path)
    if isinstance(loaded, VocabularyTokenizer):
        return loaded
    # HuggingFace tokenizer — already satisfies ChatTokenizer
    return loaded


def _load_hf_tokenizer_as_vocabulary(model_id: str) -> ChatTokenizer:
    """Load a HuggingFace AutoTokenizer, preserving BPE encoding."""
    from serve.hf_model_loader import load_huggingface_tokenizer
    from serve.huggingface_tokenizer import AutoTokenizerAdapter

    hf_tok = load_huggingface_tokenizer(model_id)
    return AutoTokenizerAdapter(hf_tok)


def validate_file_paths(**paths: str | None) -> None:
    """Validate that all provided file paths exist.

    Skips None values and HuggingFace model IDs. Raises CrucibleServeError
    with a clear message if any file path does not exist.
    """
    for name, path in paths.items():
        if not path:
            continue
        from serve.hf_model_loader import is_huggingface_model_id

        if is_huggingface_model_id(path):
            continue
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            flag = name.replace("_", "-")
            raise CrucibleServeError(
                f"File not found: {resolved} (--{flag}). "
                "Check the path and try again."
            )


def validate_training_options(options: TrainingOptions) -> None:
    """Validate training options before loop execution."""
    _validate_core_options(options)
    _validate_optimizer_options(options)
    _validate_scheduler_options(options)
    _validate_architecture_options(options)
    _validate_checkpoint_options(options)


def _validate_core_options(options: TrainingOptions) -> None:
    if options.epochs < 1:
        raise CrucibleServeError(f"Invalid epochs value {options.epochs}: expected value >= 1.")
    if options.batch_size < 1:
        raise CrucibleServeError(f"Invalid batch_size {options.batch_size}: expected value >= 1.")
    if options.max_token_length < 4:
        raise CrucibleServeError(
            f"Invalid max_token_length {options.max_token_length}: expected value >= 4."
        )
    if options.learning_rate <= 0:
        raise CrucibleServeError(
            f"Invalid learning_rate {options.learning_rate}: expected positive value."
        )
    if options.precision_mode not in SUPPORTED_TRAIN_PRECISION_MODES:
        raise CrucibleServeError(
            f"Invalid precision_mode {options.precision_mode!r}: expected one of "
            f"{', '.join(SUPPORTED_TRAIN_PRECISION_MODES)}."
        )
    if not 0 <= options.validation_split < 1:
        raise CrucibleServeError("Invalid validation_split: expected value in [0, 1).")


def _validate_optimizer_options(options: TrainingOptions) -> None:
    if options.optimizer_type not in SUPPORTED_TRAIN_OPTIMIZER_TYPES:
        raise CrucibleServeError(
            f"Invalid optimizer_type {options.optimizer_type!r}: expected one of "
            f"{', '.join(SUPPORTED_TRAIN_OPTIMIZER_TYPES)}."
        )
    if options.weight_decay < 0:
        raise CrucibleServeError(f"Invalid weight_decay {options.weight_decay}: expected value >= 0.")
    if options.sgd_momentum < 0:
        raise CrucibleServeError(f"Invalid sgd_momentum {options.sgd_momentum}: expected value >= 0.")


def _validate_scheduler_options(options: TrainingOptions) -> None:
    if options.scheduler_type not in SUPPORTED_TRAIN_SCHEDULER_TYPES:
        raise CrucibleServeError(
            f"Invalid scheduler_type {options.scheduler_type!r}: expected one of "
            f"{', '.join(SUPPORTED_TRAIN_SCHEDULER_TYPES)}."
        )
    if options.scheduler_type == "step":
        if options.scheduler_step_size < 1:
            raise CrucibleServeError(
                f"Invalid scheduler_step_size {options.scheduler_step_size}: expected value >= 1."
            )
        if not 0 < options.scheduler_gamma < 1:
            raise CrucibleServeError(
                f"Invalid scheduler_gamma {options.scheduler_gamma}: expected value in (0, 1)."
            )
    if options.scheduler_type == "cosine":
        if options.scheduler_t_max_epochs is not None and options.scheduler_t_max_epochs < 1:
            raise CrucibleServeError(
                f"Invalid scheduler_t_max_epochs {options.scheduler_t_max_epochs}: expected >= 1."
            )
        if options.scheduler_eta_min < 0:
            raise CrucibleServeError(
                f"Invalid scheduler_eta_min {options.scheduler_eta_min}: expected value >= 0."
            )


def _validate_architecture_options(options: TrainingOptions) -> None:
    if options.hidden_dim < 1:
        raise CrucibleServeError(f"Invalid hidden_dim {options.hidden_dim}: expected value >= 1.")
    if options.num_layers < 1:
        raise CrucibleServeError(f"Invalid num_layers {options.num_layers}: expected value >= 1.")
    if options.attention_heads < 1:
        raise CrucibleServeError(
            f"Invalid attention_heads {options.attention_heads}: expected value >= 1."
        )
    if options.hidden_dim % options.attention_heads != 0:
        raise CrucibleServeError(
            f"Invalid configuration: hidden_dim {options.hidden_dim} must be divisible by "
            f"attention_heads {options.attention_heads}."
        )
    if options.mlp_hidden_dim < 1:
        raise CrucibleServeError(
            f"Invalid mlp_hidden_dim {options.mlp_hidden_dim}: expected value >= 1."
        )
    if options.mlp_layers < 1:
        raise CrucibleServeError(f"Invalid mlp_layers {options.mlp_layers}: expected value >= 1.")
    if not 0 <= options.dropout < 1:
        raise CrucibleServeError(f"Invalid dropout {options.dropout}: expected value in [0, 1).")
    if options.position_embedding_type not in {"learned", "sinusoidal"}:
        raise CrucibleServeError(
            "Invalid position_embedding_type "
            f"{options.position_embedding_type!r}: expected 'learned' or 'sinusoidal'."
        )
    if options.vocabulary_size is not None and options.vocabulary_size < 2:
        raise CrucibleServeError(
            f"Invalid vocabulary_size {options.vocabulary_size}: expected value >= 2."
        )


def _validate_checkpoint_options(options: TrainingOptions) -> None:
    if options.checkpoint_every_epochs < 1:
        raise CrucibleServeError(
            "Invalid checkpoint_every_epochs "
            f"{options.checkpoint_every_epochs}: expected value >= 1."
        )
    if options.max_checkpoint_files is not None and options.max_checkpoint_files < 1:
        raise CrucibleServeError(
            f"Invalid max_checkpoint_files {options.max_checkpoint_files}: expected >= 1."
        )
    if options.progress_log_interval_steps < 1:
        raise CrucibleServeError(
            "Invalid progress_log_interval_steps "
            f"{options.progress_log_interval_steps}: expected value >= 1."
        )
    if options.initial_weights_path and options.resume_checkpoint_path:
        raise CrucibleServeError(
            "Invalid configuration: initial_weights_path and resume_checkpoint_path are "
            "mutually exclusive. Use resume_checkpoint_path to continue a prior run or "
            "initial_weights_path to start fine-tuning from model weights."
        )
