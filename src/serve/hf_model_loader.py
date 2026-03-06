"""HuggingFace model loading utilities.

Centralizes loading models from HuggingFace model IDs with optional
custom weight overrides. Used by chat, LoRA training, and other runners
that need to work with pretrained model architectures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.errors import ForgeDependencyError, ForgeServeError


def is_huggingface_model_id(model_path: str) -> bool:
    """Check if the path looks like a HuggingFace model ID or local HF directory.

    Returns True for model IDs like 'gpt2', 'meta-llama/Llama-2-7b'.
    Returns True for local directories containing config.json (downloaded HF models).
    Returns False for file paths like '/path/to/model.pt'.
    """
    path = Path(model_path)
    if path.is_dir():
        return (path / "config.json").exists()
    if path.exists():
        return False
    if path.suffix in (".pt", ".bin", ".safetensors", ".onnx"):
        return False
    return True


def load_huggingface_model(
    model_id: str,
    weights_path: str | None = None,
    device: Any = None,
) -> Any:
    """Load a HuggingFace model with optional custom weights.

    Args:
        model_id: HuggingFace model identifier (e.g. 'gpt2').
        weights_path: Optional path to custom weights (.pt, .safetensors).
            If None, uses the pretrained weights from HuggingFace.
        device: Target device. If None, stays on CPU.

    Returns:
        Loaded PyTorch model.

    Raises:
        ForgeDependencyError: If transformers is not installed.
        ForgeServeError: If model loading fails.
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as error:
        raise ForgeDependencyError(
            "Loading HuggingFace models requires transformers. "
            "Install with: pip install transformers"
        ) from error

    # Check if model uses quantization that requires CUDA (FP8, etc.)
    # and force CPU if we're on a non-CUDA device.
    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "dtype": "auto",
    }
    try:
        config = AutoConfig.from_pretrained(model_id)
        quant_config = getattr(config, "quantization_config", None)
        if quant_config:
            quant_type = ""
            if isinstance(quant_config, dict):
                quant_type = quant_config.get("quant_method", "")
            else:
                quant_type = getattr(quant_config, "quant_method", "")
            _MPS_INCOMPATIBLE_QUANT = {"compressed-tensors", "fp8", "fbgemm_fp8"}
            if quant_type in _MPS_INCOMPATIBLE_QUANT:
                import torch
                if not torch.cuda.is_available():
                    load_kwargs["device_map"] = "cpu"
    except Exception:
        pass

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except ImportError as error:
        raise ForgeDependencyError(
            f"Model '{model_id}' requires an extra dependency: {error}. "
            "Install it and try again."
        ) from error
    except Exception as error:
        raise ForgeServeError(
            f"Failed to load HuggingFace model '{model_id}': {error}. "
            "Verify the model ID is correct (e.g. 'gpt2', 'meta-llama/Llama-2-7b')."
        ) from error

    if weights_path:
        _load_custom_weights(model, weights_path)

    # device_map="auto" handles placement; only move if no device_map
    if device is not None and not getattr(model, "hf_device_map", None):
        model = model.to(device)

    return model


def load_huggingface_tokenizer(model_id: str) -> Any:
    """Load a HuggingFace tokenizer by model ID.

    Returns the tokenizer with pad_token set if missing.

    Raises:
        ForgeDependencyError: If transformers is not installed.
        ForgeServeError: If tokenizer loading fails.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ForgeDependencyError(
            "Loading HuggingFace tokenizers requires transformers. "
            "Install with: pip install transformers"
        ) from error

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as error:
        raise ForgeServeError(
            f"Failed to load tokenizer for '{model_id}': {error}."
        ) from error

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def _make_logits_wrapper(hf_model: Any) -> Any:
    """Wrap a HuggingFace causal LM so forward() returns raw logits.

    The Forge training loop calls model(inputs) and expects a plain
    tensor of logits. HuggingFace models return dataclass-like objects
    with a .logits attribute. This creates a thin nn.Module wrapper
    that bridges the two interfaces.
    """
    try:
        import torch.nn as nn
    except ImportError as error:
        raise ForgeDependencyError("torch is required") from error

    class _HfLogitsWrapper(nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.model = model

        def forward(self, input_ids: Any) -> Any:
            outputs = self.model(input_ids=input_ids)
            return outputs.logits

    return _HfLogitsWrapper(hf_model)


def build_or_load_model(
    torch_module: Any,
    base_model: str | None,
    build_forge_model: Any,
    device: Any,
) -> Any:
    """Build a Forge model or load a HuggingFace model depending on base_model.

    If base_model is a HuggingFace model ID (e.g. 'gpt2'), loads the full
    pretrained model via transformers and wraps it so forward() returns raw
    logits (compatible with Forge training loops). Otherwise falls back to
    the provided build_forge_model callable.

    Args:
        torch_module: Imported torch module.
        base_model: HuggingFace model ID, file path, or None.
        build_forge_model: Callable that returns a Forge model when no HF
            model is used. Signature: () -> model.
        device: Target device for the model.

    Returns:
        The loaded or built model on the target device.
    """
    if base_model and is_huggingface_model_id(base_model):
        model = load_huggingface_model(base_model, device=device)
        return _make_logits_wrapper(model)

    model = build_forge_model()
    model = model.to(device)
    return model


def _load_custom_weights(model: Any, weights_path: str) -> None:
    """Load custom weights into an existing model architecture."""
    resolved = Path(weights_path).expanduser().resolve()
    if not resolved.exists():
        raise ForgeServeError(
            f"Custom weights file not found: {resolved}. "
            "Verify the path is correct."
        )

    try:
        import torch
    except ImportError as error:
        raise ForgeDependencyError(
            "Loading custom weights requires torch."
        ) from error

    suffix = resolved.suffix.lower()
    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(resolved))
        except ImportError:
            raise ForgeDependencyError(
                "Loading .safetensors weights requires the safetensors library. "
                "Install with: pip install safetensors"
            )
    elif suffix in (".pt", ".pth", ".bin"):
        state_dict = torch.load(str(resolved), map_location="cpu")
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
    else:
        raise ForgeServeError(
            f"Unsupported weight format '{suffix}' for {resolved}. "
            "Supported: .pt, .pth, .bin, .safetensors"
        )

    model.load_state_dict(state_dict, strict=False)
