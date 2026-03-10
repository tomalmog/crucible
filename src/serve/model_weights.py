"""Model weight loading utilities for training runs.

This module applies optional initial model weights for fine-tuning.
It keeps checkpoint parsing and error reporting separate from loop logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.errors import CrucibleDependencyError, CrucibleServeError
from serve.model_format import detect_model_format


def load_initial_weights(
    torch_module: Any,
    model: Any,
    initial_weights_path: str | None,
    device: Any,
) -> None:
    """Load optional initial model weights into the training model.

    Args:
        torch_module: Imported torch module.
        model: Model instance receiving state dict.
        initial_weights_path: Optional checkpoint path.
        device: Resolved training device.

    Raises:
        CrucibleServeError: If weights cannot be read or applied.
    """
    if initial_weights_path is None:
        return
    resolved_path = Path(initial_weights_path).expanduser().resolve()
    model_state = read_model_state_dict(torch_module, str(resolved_path), device)
    _apply_model_state(model, model_state, resolved_path)


def read_model_state_dict(
    torch_module: Any,
    weights_path: str,
    device: Any,
) -> Mapping[str, object]:
    """Read and normalize a model state dict from a checkpoint file.

    Args:
        torch_module: Imported torch module.
        weights_path: Path to checkpoint file.
        device: Resolved torch device.

    Returns:
        Parsed model state dictionary.

    Raises:
        CrucibleServeError: If checkpoint cannot be read or parsed.
    """
    resolved_path = Path(weights_path).expanduser().resolve()
    if not resolved_path.exists():
        raise CrucibleServeError(
            f"Initial weights not found at {resolved_path}. "
            "Provide a valid --initial-weights-path or omit it to train from scratch."
        )
    # If path is a directory, find the model weights file inside it
    if resolved_path.is_dir():
        if (resolved_path / "config.json").exists():
            raise CrucibleServeError(
                f"Path {resolved_path} is a HuggingFace model directory. "
                "Use a HuggingFace model ID or --base-model-path instead of --initial-weights-path."
            )
        resolved_path = _find_weights_file_in_dir(resolved_path)
    if detect_model_format(str(resolved_path)) == "onnx":
        return _read_onnx_model_state_dict(torch_module, resolved_path, device)
    checkpoint_payload = _read_torch_checkpoint(torch_module, resolved_path, device)
    return _extract_model_state(checkpoint_payload, resolved_path)


def _find_weights_file_in_dir(directory: Path) -> Path:
    """Find a model weights file inside a directory.

    Checks for common weight file names in priority order.

    Raises:
        CrucibleServeError: If no recognized weights file is found.
    """
    candidates = [
        "model.pt",
        "model.safetensors",
        "pytorch_model.bin",
        "model.pth",
        "model.bin",
        "model.onnx",
    ]
    for name in candidates:
        path = directory / name
        if path.exists():
            return path
    found = [
        f"{f.name}/" if f.is_dir() else f.name
        for f in sorted(directory.iterdir())
    ]
    raise CrucibleServeError(
        f"Path {directory} is a directory but contains no recognized model weights. "
        f"Expected one of: {', '.join(candidates)}. "
        f"Directory contents: {', '.join(found) or '(empty)'}"
    )


def _read_torch_checkpoint(torch_module: Any, weights_path: Path, device: Any) -> object:
    """Read torch checkpoint payload with consistent error handling."""
    try:
        return torch_module.load(str(weights_path), map_location=device)
    except (OSError, RuntimeError) as error:
        raise CrucibleServeError(
            f"Failed to load initial weights from {weights_path}: {error}. "
            "Verify the checkpoint file is readable and valid."
        ) from error


def _read_onnx_model_state_dict(
    torch_module: Any,
    weights_path: Path,
    device: Any,
) -> Mapping[str, object]:
    """Read ONNX initializer tensors as a torch-compatible state dict."""
    onnx_module = _import_onnx_optional()
    numpy_helper = getattr(onnx_module, "numpy_helper", None)
    if numpy_helper is None:
        raise CrucibleServeError(
            f"Failed to read ONNX initializers from {weights_path}: numpy_helper is missing."
        )
    try:
        onnx_model = onnx_module.load(str(weights_path))
    except Exception as error:
        raise CrucibleServeError(
            f"Failed to load ONNX model weights from {weights_path}: {error}. "
            "Verify the ONNX file is readable and valid."
        ) from error
    return _extract_onnx_initializers(torch_module, onnx_model, numpy_helper, weights_path, device)


def _extract_onnx_initializers(
    torch_module: Any,
    onnx_model: Any,
    numpy_helper: Any,
    weights_path: Path,
    device: Any,
) -> Mapping[str, object]:
    """Convert ONNX initializer payload into model state mapping."""
    graph = getattr(onnx_model, "graph", None)
    initializers = list(getattr(graph, "initializer", []))
    if not initializers:
        raise CrucibleServeError(
            f"Invalid ONNX weights at {weights_path}: graph contains no initializer tensors."
        )
    state_dict: dict[str, object] = {}
    for initializer in initializers:
        parameter_name = str(getattr(initializer, "name", ""))
        if not parameter_name:
            continue
        parameter_array = numpy_helper.to_array(initializer)
        state_dict[parameter_name] = torch_module.tensor(parameter_array, device=device)
    if not state_dict:
        raise CrucibleServeError(
            f"Invalid ONNX weights at {weights_path}: no named initializer tensors found."
        )
    return state_dict


def _import_onnx_optional() -> Any:
    """Import ONNX dependency used for ONNX initial-weights loading."""
    try:
        import onnx
    except ImportError as error:
        raise CrucibleDependencyError(
            "Loading ONNX initial weights requires onnx, but it is not installed. "
            "Install with pip install -e .[onnx] before using --initial-weights-path on .onnx."
        ) from error
    return onnx


def _apply_model_state(model: Any, model_state: Mapping[str, object], weights_path: Path) -> None:
    """Apply checkpoint state dict to model with traceable error handling."""
    try:
        model.load_state_dict(model_state)
    except RuntimeError as error:
        raise CrucibleServeError(
            f"Failed to apply initial weights from {weights_path}: {error}. "
            "Use matching architecture settings or train from scratch."
        ) from error


def _extract_model_state(checkpoint_payload: object, weights_path: Path) -> Mapping[str, object]:
    """Extract model state dict from raw checkpoint payload."""
    if isinstance(checkpoint_payload, Mapping):
        nested_payload = checkpoint_payload.get("model_state_dict")
        if isinstance(nested_payload, Mapping):
            return nested_payload
        if all(isinstance(key, str) for key in checkpoint_payload):
            return checkpoint_payload
    raise CrucibleServeError(
        f"Invalid checkpoint format at {weights_path}: expected a state_dict mapping "
        "or a mapping containing model_state_dict."
    )
