"""ONNX Runtime chat inference runner.

This module executes autoregressive text generation against ONNX model
artifacts exported from compatible Crucible/PyTorch architectures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.chat_types import ChatOptions, ChatTokenizer
from core.errors import CrucibleDependencyError, CrucibleServeError
from core.types import DataRecord
from serve.chat_option_resolver import resolve_chat_tokenizer, resolve_chat_training_options


@dataclass
class OnnxChatContext:
    """Runtime objects required for ONNX chat generation."""

    session: Any
    np_module: Any
    input_names: list[str]
    output_name: str
    tokenizer: ChatTokenizer
    options: ChatOptions
    max_context_tokens: int


def run_onnx_chat(records: list[DataRecord] | None, options: ChatOptions) -> str:
    """Run one ONNX Runtime text generation call."""
    context = _build_onnx_chat_context(records, options)
    return _generate_response_text(context)


def _build_onnx_chat_context(
    records: list[DataRecord] | None,
    options: ChatOptions,
) -> OnnxChatContext:
    model_path = _resolve_model_path(options.model_path)
    ort_module = _import_onnxruntime_optional()
    np_module = _import_numpy_optional()
    session = _build_session(ort_module, model_path)
    input_names = _resolve_input_names(session)
    output_name = _resolve_output_name(session)
    training_options = resolve_chat_training_options(options, model_state={})
    tokenizer = resolve_chat_tokenizer(records, options, training_options)
    return OnnxChatContext(
        session=session,
        np_module=np_module,
        input_names=input_names,
        output_name=output_name,
        tokenizer=tokenizer,
        options=options,
        max_context_tokens=training_options.max_token_length,
    )


def _resolve_model_path(model_path: str) -> Path:
    resolved_path = Path(model_path).expanduser().resolve()
    if not resolved_path.exists():
        raise CrucibleServeError(
            f"Model file not found at {resolved_path}. "
            "Provide a valid --model-path for chat inference."
        )
    return resolved_path


def _import_onnxruntime_optional() -> Any:
    try:
        import onnxruntime
    except ImportError as error:
        raise CrucibleDependencyError(
            "ONNX inference requires onnxruntime, but it is not installed. "
            "Install with pip install -e .[onnx] to run chat on .onnx models."
        ) from error
    return onnxruntime


def _import_numpy_optional() -> Any:
    try:
        import numpy
    except ImportError as error:
        raise CrucibleDependencyError(
            "ONNX inference requires numpy, but it is not installed. "
            "Install with pip install -e .[onnx] to run chat on .onnx models."
        ) from error
    return numpy


def _build_session(ort_module: Any, model_path: Path) -> Any:
    providers = list(ort_module.get_available_providers())
    try:
        return ort_module.InferenceSession(str(model_path), providers=providers)
    except Exception as error:
        raise CrucibleServeError(
            f"Failed to initialize ONNX Runtime session from {model_path}: {error}. "
            "Verify the ONNX artifact and retry."
        ) from error


def _resolve_input_names(session: Any) -> list[str]:
    """Resolve all input names declared by the ONNX model graph."""
    inputs = list(session.get_inputs())
    if not inputs:
        raise CrucibleServeError("ONNX model has no inputs. Provide a valid language-model graph.")
    return [str(inp.name) for inp in inputs]


def _resolve_output_name(session: Any) -> str:
    outputs = list(session.get_outputs())
    if not outputs:
        raise CrucibleServeError("ONNX model has no outputs. Provide a valid language-model graph.")
    return str(outputs[0].name)


def _generate_response_text(context: OnnxChatContext) -> str:
    import sys

    options = context.options
    prompt_ids = context.tokenizer.encode(options.prompt, context.max_context_tokens)
    context_ids = prompt_ids if prompt_ids else [1]
    generated_ids: list[int] = []
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
    return str(context.tokenizer.decode(generated_ids)).strip()


def _build_input_feed(context: OnnxChatContext, input_tensor: Any) -> dict[str, Any]:
    """Build ONNX input feed from model-declared input names.

    Always provides input_ids. Generates attention_mask (all ones) and
    position_ids (sequential) when the model graph requires them.
    """
    np_module = context.np_module
    feed: dict[str, Any] = {}
    for name in context.input_names:
        if name == "input_ids":
            feed[name] = input_tensor
        elif name == "attention_mask":
            feed[name] = np_module.ones_like(input_tensor)
        elif name == "position_ids":
            seq_length = int(input_tensor.shape[1])
            feed[name] = np_module.arange(seq_length, dtype=np_module.int64).reshape(1, -1)
    return feed


def _sample_next_token(context: OnnxChatContext, context_ids: list[int]) -> int:
    np_module = context.np_module
    input_ids = context_ids[-context.max_context_tokens :]
    input_tensor = np_module.asarray([input_ids], dtype=np_module.int64)
    input_feed = _build_input_feed(context, input_tensor)
    outputs = context.session.run([context.output_name], input_feed)
    logits = outputs[0]
    next_logits = logits[0, -1, :].astype(np_module.float64)
    temperature = context.options.temperature
    if temperature == 0:
        return int(np_module.argmax(next_logits))
    scaled_logits = next_logits / temperature
    if context.options.top_k > 0:
        return _sample_top_k(np_module, scaled_logits, context.options.top_k)
    probabilities = _softmax(np_module, scaled_logits)
    return int(np_module.random.choice(len(probabilities), p=probabilities))


def _sample_top_k(np_module: Any, scaled_logits: Any, requested_top_k: int) -> int:
    vocabulary_size = int(scaled_logits.shape[-1])
    top_k = min(requested_top_k, vocabulary_size)
    top_indexes = np_module.argpartition(scaled_logits, -top_k)[-top_k:]
    top_logits = scaled_logits[top_indexes]
    probabilities = _softmax(np_module, top_logits)
    sampled_index = int(np_module.random.choice(len(top_indexes), p=probabilities))
    return int(top_indexes[sampled_index])


def _softmax(np_module: Any, logits: Any) -> Any:
    shifted_logits = logits - np_module.max(logits)
    exponentials = np_module.exp(shifted_logits)
    return exponentials / np_module.sum(exponentials)
