"""PyTorch chat inference runner for Crucible-trained models.

This module loads a trained model checkpoint and generates text
responses from prompt input using training-compatible settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.chat_types import ChatOptions, ChatResult, ChatTokenizer
from core.errors import CrucibleDependencyError, CrucibleServeError
from core.types import DataRecord
from serve.architecture_loader import load_training_model
from serve.chat_option_resolver import (
    resolve_chat_model_vocab_size,
    resolve_chat_tokenizer,
    resolve_chat_training_options,
)
from serve.device_selection import resolve_execution_device
from serve.hf_model_loader import is_huggingface_model_id, load_huggingface_model, load_huggingface_tokenizer
from serve.model_format import detect_model_format
from serve.model_weights import load_initial_weights, read_model_state_dict
from serve.onnx_chat_runner import run_onnx_chat
from serve.training_setup import validate_training_options


@dataclass
class ChatRuntimeContext:
    """Runtime objects needed for chat text generation."""

    torch_module: Any
    model: Any
    tokenizer: ChatTokenizer
    options: ChatOptions
    device: Any
    max_context_tokens: int


def run_chat(records: list[DataRecord] | None, options: ChatOptions) -> ChatResult:
    """Run one model chat inference call.

    Args:
        records: Dataset records for tokenizer fallback, or None when a
            persisted or explicit tokenizer is available.
        options: Chat inference options.

    Returns:
        Generated response text.

    Raises:
        CrucibleServeError: If model loading or generation fails.
    """
    _validate_chat_options(options)
    if is_huggingface_model_id(options.model_path):
        context = _build_hf_runtime_context(options)
        response_text = _generate_response_text(context)
        return ChatResult(response_text=response_text)
    # Check if this is a merged HF model (e.g. from LoRA training on a HF base).
    # The training_config.json will have base_model_path pointing to the HF model.
    hf_base = _detect_hf_base_model(options.model_path)
    if hf_base:
        context = _build_hf_runtime_context(options, hf_model_id=hf_base)
        response_text = _generate_response_text(context)
        return ChatResult(response_text=response_text)
    model_format = _detect_dir_aware_format(options.model_path)
    if model_format == "onnx":
        response_text = run_onnx_chat(records, options)
        return ChatResult(response_text=response_text)
    context = _build_runtime_context(records, options)
    response_text = _generate_response_text(context)
    return ChatResult(response_text=response_text)


def _build_hf_runtime_context(
    options: ChatOptions,
    hf_model_id: str | None = None,
) -> ChatRuntimeContext:
    """Build chat runtime using a HuggingFace model.

    Args:
        options: Chat options. When hf_model_id is None, options.model_path
            is used as the HuggingFace model ID.
        hf_model_id: Override HF model ID (used for merged LoRA models where
            model_path is a .pt file but the architecture comes from a HF base).
    """
    torch_module = _import_torch()
    device = _resolve_inference_device(torch_module)

    model_id = hf_model_id or options.model_path
    weights_path = options.model_path if hf_model_id else options.weights_path
    model = load_huggingface_model(model_id, weights_path, device)
    model.eval()

    # Detect actual device the model landed on (device_map may override)
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        first_device = next(iter(device_map.values()))
        device = torch_module.device(first_device)

    # Load tokenizer
    hf_tokenizer = load_huggingface_tokenizer(model_id)
    tokenizer = _HfTokenizerAdapter(hf_tokenizer)

    max_context = getattr(model.config, "n_positions", None) or options.max_token_length
    return ChatRuntimeContext(
        torch_module=torch_module,
        model=model,
        tokenizer=tokenizer,
        options=options,
        device=device,
        max_context_tokens=max_context,
    )


class _HfTokenizerAdapter:
    """Adapts a HuggingFace tokenizer to the ChatTokenizer protocol."""

    def __init__(self, hf_tokenizer: Any) -> None:
        self._tokenizer = hf_tokenizer
        vocab = hf_tokenizer.get_vocab()
        self.vocabulary: dict[str, int] = dict(vocab)

    def encode(self, text: str, max_token_length: int) -> list[int]:
        ids = self._tokenizer.encode(text)
        return ids[:max_token_length]

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids)


def _build_runtime_context(
    records: list[DataRecord] | None,
    options: ChatOptions,
) -> ChatRuntimeContext:
    """Build chat runtime context from dataset records and options."""
    torch_module = _import_torch()
    device = _resolve_inference_device(torch_module)
    model_state = read_model_state_dict(torch_module, options.model_path, device)
    training_options = resolve_chat_training_options(options, model_state)
    validate_training_options(training_options)
    tokenizer = resolve_chat_tokenizer(records, options, training_options)
    model = load_training_model(
        torch_module,
        training_options,
        resolve_chat_model_vocab_size(tokenizer.vocabulary, model_state, training_options),
    )
    model = model.to(device)
    load_initial_weights(
        torch_module=torch_module,
        model=model,
        initial_weights_path=options.model_path,
        device=device,
    )
    model.eval()
    max_context_tokens = _resolve_runtime_context_limit(model, training_options.max_token_length)
    return ChatRuntimeContext(
        torch_module=torch_module,
        model=model,
        tokenizer=tokenizer,
        options=options,
        device=device,
        max_context_tokens=max_context_tokens,
    )


def _detect_hf_base_model(model_path: str) -> str | None:
    """Check if model_path has a training_config with a HuggingFace base_model_path.

    Returns the base model ID if found, None otherwise.
    """
    from serve.training_metadata import load_training_config

    config = load_training_config(model_path)
    if not config:
        return None
    base = str(config.get("base_model_path", "") or "")
    if base and is_huggingface_model_id(base):
        return base
    return None


def _import_torch() -> Any:
    """Import torch dependency."""
    try:
        import torch
    except ImportError as error:
        raise CrucibleDependencyError(
            "Chat inference requires torch, but it is not installed. Install torch to run crucible chat."
        ) from error
    return torch


def _detect_dir_aware_format(model_path: str) -> str:
    """Detect model format, searching inside directories for weight files."""
    from pathlib import Path as _Path

    path = _Path(model_path).expanduser().resolve()
    if path.is_dir():
        for name in ("model.onnx", "model.pt", "model.safetensors",
                      "pytorch_model.bin", "model.pth", "model.bin"):
            if (path / name).exists():
                return detect_model_format(str(path / name))
        return "unknown"
    return detect_model_format(model_path)


def _resolve_inference_device(torch_module: Any) -> Any:
    """Select torch device for inference."""
    return resolve_execution_device(torch_module)


def _validate_chat_options(options: ChatOptions) -> None:
    """Validate chat-specific option fields."""
    if not options.prompt.strip():
        raise CrucibleServeError(
            "Invalid prompt: expected non-empty input text. Provide --prompt with message content."
        )
    if options.max_new_tokens < 1:
        raise CrucibleServeError(
            f"Invalid max_new_tokens {options.max_new_tokens}: expected value >= 1."
        )
    if options.temperature < 0:
        raise CrucibleServeError(f"Invalid temperature {options.temperature}: expected value >= 0.")
    if options.top_k < 0:
        raise CrucibleServeError(f"Invalid top_k {options.top_k}: expected value >= 0.")


def _generate_response_text(context: ChatRuntimeContext) -> str:
    """Generate response text from prompt with autoregressive decoding."""
    import sys

    options = context.options
    prompt_ids = context.tokenizer.encode(options.prompt, context.max_context_tokens)
    if not prompt_ids:
        prompt_ids = [1]
    generated_ids: list[int] = []
    context_ids = list(prompt_ids)
    for _ in range(options.max_new_tokens):
        next_token_id = _sample_next_token(context, context_ids)
        if next_token_id == 0:
            break
        context_ids.append(next_token_id)
        generated_ids.append(next_token_id)
        if options.stream:
            token_text = context.tokenizer.decode([next_token_id])
            sys.stdout.write(token_text)
            sys.stdout.flush()
    if not generated_ids:
        return ""
    decoded = context.tokenizer.decode(generated_ids)
    return decoded.strip()


def _sample_next_token(context: ChatRuntimeContext, context_ids: list[int]) -> int:
    """Sample the next token id using configured decoding settings."""
    torch_module = context.torch_module
    options = context.options
    input_ids = context_ids[-context.max_context_tokens :]
    input_tensor = torch_module.tensor([input_ids], dtype=torch_module.long).to(context.device)
    with torch_module.no_grad():
        output = context.model(input_tensor)
    logits = output.logits if hasattr(output, "logits") else output
    next_logits = logits[0, -1, :]
    if options.temperature == 0:
        return int(torch_module.argmax(next_logits).item())
    scaled_logits = next_logits / options.temperature
    if options.top_k > 0:
        top_k = min(options.top_k, int(scaled_logits.shape[-1]))
        values, indices = torch_module.topk(scaled_logits, top_k)
        probabilities = torch_module.softmax(values, dim=-1)
        sampled_position = int(torch_module.multinomial(probabilities, num_samples=1).item())
        return int(indices[sampled_position].item())
    probabilities = torch_module.softmax(scaled_logits, dim=-1)
    return int(torch_module.multinomial(probabilities, num_samples=1).item())


def _resolve_runtime_context_limit(model: Any, fallback_limit: int) -> int:
    """Resolve the usable max context length for inference."""
    position_embedding = getattr(model, "position_embedding", None)
    if position_embedding is not None:
        num_embeddings = getattr(position_embedding, "num_embeddings", None)
        if isinstance(num_embeddings, int) and num_embeddings > 0:
            return num_embeddings
    sinusoidal = getattr(model, "sinusoidal_position_encoding", None)
    if sinusoidal is not None:
        shape = getattr(sinusoidal, "shape", None)
        if shape is not None and len(shape) > 0 and int(shape[0]) > 0:
            return int(shape[0])
    return fallback_limit
