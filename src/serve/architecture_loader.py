"""Model architecture loader for training runs.

This module supports architecture definition via Python or JSON files.
It enables overriding the default PyTorch training model.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any, cast

from core.errors import CrucibleServeError
from core.types import PositionEmbeddingType, TrainingOptions
from serve.default_model import build_default_model


def load_training_model(
    torch_module: Any,
    options: TrainingOptions,
    vocab_size: int,
) -> Any:
    """Load model from architecture config or default implementation.

    Args:
        torch_module: Imported torch module.
        options: Training options.
        vocab_size: Token vocabulary size.

    Returns:
        torch.nn.Module training model.

    Raises:
        CrucibleServeError: If architecture file is invalid.
    """
    if options.architecture_path is None:
        return build_default_model(
            torch_module=torch_module,
            vocab_size=vocab_size,
            options=options,
        )
    architecture_path = Path(options.architecture_path).expanduser().resolve()
    if not architecture_path.exists():
        raise CrucibleServeError(
            f"Architecture file not found at {architecture_path}. "
            "Provide a valid --architecture-file path."
        )
    suffix = architecture_path.suffix.lower()
    if suffix == ".py":
        return _load_model_from_python(torch_module, architecture_path, options, vocab_size)
    if suffix == ".json":
        return _load_model_from_json(torch_module, architecture_path, options, vocab_size)
    raise CrucibleServeError(
        f"Unsupported architecture format '{suffix}' at {architecture_path}. "
        "Use .py or .json architecture files."
    )


def _load_model_from_python(
    torch_module: Any,
    architecture_path: Path,
    options: TrainingOptions,
    vocab_size: int,
) -> Any:
    """Load model builder function from Python file."""
    module = _load_python_module(architecture_path)
    builder = getattr(module, "build_model", None)
    if builder is None or not callable(builder):
        raise CrucibleServeError(
            f"Invalid architecture file at {architecture_path}: missing callable build_model. "
            "Define build_model(vocab_size, torch_module[, options]) in the file."
        )
    builder_signature = inspect.signature(builder)
    builder_kwargs: dict[str, object] = {
        "vocab_size": vocab_size,
        "torch_module": torch_module,
    }
    if _accepts_options_argument(builder_signature):
        builder_kwargs["options"] = options
    model = builder(**builder_kwargs)
    _validate_model_instance(torch_module, model, architecture_path)
    return model


def _load_model_from_json(
    torch_module: Any,
    architecture_path: Path,
    options: TrainingOptions,
    vocab_size: int,
) -> Any:
    """Load architecture from JSON config."""
    payload = _read_json_file(architecture_path)
    architecture_type = str(payload.get("architecture", "default"))
    if architecture_type != "default":
        raise CrucibleServeError(
            f"Unsupported architecture '{architecture_type}' in {architecture_path}. "
            "Set architecture to 'default' for JSON-based configs."
        )
    training_options = TrainingOptions(
        dataset_name=options.dataset_name,
        output_dir=options.output_dir,
        architecture_path=options.architecture_path,
        custom_loop_path=options.custom_loop_path,
        epochs=options.epochs,
        learning_rate=options.learning_rate,
        batch_size=options.batch_size,
        max_token_length=options.max_token_length,
        validation_split=options.validation_split,
        hidden_dim=_read_config_int(payload, "hidden_dim", options.hidden_dim, architecture_path),
        num_layers=_read_config_int(payload, "num_layers", options.num_layers, architecture_path),
        attention_heads=_read_config_int(
            payload,
            "attention_heads",
            options.attention_heads,
            architecture_path,
        ),
        mlp_hidden_dim=_read_config_int(
            payload,
            "mlp_hidden_dim",
            options.mlp_hidden_dim,
            architecture_path,
        ),
        mlp_layers=_read_config_int(payload, "mlp_layers", options.mlp_layers, architecture_path),
        dropout=_read_config_float(payload, "dropout", options.dropout, architecture_path),
        position_embedding_type=_read_position_embedding_type(
            payload,
            "position_embedding_type",
            options.position_embedding_type,
            architecture_path,
        ),
        vocabulary_size=options.vocabulary_size,
        initial_weights_path=options.initial_weights_path,
    )
    return build_default_model(
        torch_module=torch_module,
        vocab_size=vocab_size,
        options=training_options,
    )


def _load_python_module(module_path: Path) -> Any:
    """Load Python module from file path."""
    spec = importlib.util.spec_from_file_location("crucible_user_architecture", str(module_path))
    if spec is None or spec.loader is None:
        raise CrucibleServeError(
            f"Failed to load architecture module at {module_path}. Verify the file path and syntax."
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json_file(file_path: Path) -> dict[str, object]:
    """Read JSON architecture config file."""
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CrucibleServeError(
            f"Failed to parse architecture JSON at {file_path}: {error.msg}. "
            "Fix the JSON file and retry training."
        ) from error
    if not isinstance(payload, dict):
        raise CrucibleServeError(f"Invalid architecture JSON at {file_path}: expected JSON object.")
    return payload


def _validate_model_instance(torch_module: Any, model: object, architecture_path: Path) -> None:
    """Validate that loaded model is a torch module instance."""
    if not isinstance(model, torch_module.nn.Module):
        raise CrucibleServeError(
            f"Invalid build_model result in {architecture_path}: expected torch.nn.Module. "
            "Return a torch.nn.Module instance from build_model."
        )


def _read_config_int(
    payload: dict[str, object],
    field_name: str,
    default_value: int,
    architecture_path: Path,
) -> int:
    """Read integer field from architecture payload."""
    raw_value = payload.get(field_name, default_value)
    if isinstance(raw_value, int):
        return raw_value
    raise CrucibleServeError(
        f"Invalid architecture field '{field_name}' in {architecture_path}: "
        f"expected integer, got {type(raw_value).__name__}."
    )


def _read_config_float(
    payload: dict[str, object],
    field_name: str,
    default_value: float,
    architecture_path: Path,
) -> float:
    """Read float field from architecture payload."""
    raw_value = payload.get(field_name, default_value)
    if isinstance(raw_value, (float, int)):
        return float(raw_value)
    raise CrucibleServeError(
        f"Invalid architecture field '{field_name}' in {architecture_path}: "
        f"expected float, got {type(raw_value).__name__}."
    )


def _accepts_options_argument(builder_signature: inspect.Signature) -> bool:
    """Check whether build_model can receive an options argument."""
    for parameter in builder_signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return "options" in builder_signature.parameters


def _read_position_embedding_type(
    payload: dict[str, object],
    field_name: str,
    default_value: PositionEmbeddingType,
    architecture_path: Path,
) -> PositionEmbeddingType:
    """Read enumerated string field from architecture payload."""
    supported_values = ("learned", "sinusoidal")
    raw_value = payload.get(field_name, default_value)
    if isinstance(raw_value, str) and raw_value in supported_values:
        return cast(PositionEmbeddingType, raw_value)
    supported_text = ", ".join(supported_values)
    raise CrucibleServeError(
        f"Invalid architecture field '{field_name}' in {architecture_path}: "
        f"expected one of [{supported_text}], got {raw_value!r}."
    )
