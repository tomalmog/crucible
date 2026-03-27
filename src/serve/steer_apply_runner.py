"""Steering vector application: generate text with and without steering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from core.steering_types import SteerApplyOptions
from serve.activation_extractor import discover_transformer_layers
from serve.steering_vector_io import load_steering_vector


def run_steer_apply(options: SteerApplyOptions) -> dict[str, Any]:
    """Generate text with and without a steering vector applied."""
    model, tokenizer = _load_model_and_tokenizer(options)
    model.eval()

    vector_path = Path(options.steering_vector_path).expanduser().resolve()
    device_str = str(next(model.parameters()).device)
    steering_vector, vec_meta = load_steering_vector(vector_path, device=device_str)

    target_layer = vec_meta["layer_name"]
    all_layers = discover_transformer_layers(model)
    if target_layer not in all_layers:
        layer_idx = vec_meta.get("layer_index", len(all_layers) - 1)
        target_layer = all_layers[layer_idx]

    device = next(model.parameters()).device
    ids = tokenizer.encode(
        options.input_text, return_tensors="pt", truncation=True, max_length=512,
    )
    ids = ids.to(device)

    # Generate without steering
    original_ids = _generate_tokens(model, tokenizer, ids, options.max_new_tokens)
    original_text = _decode_generated(tokenizer, original_ids, len(ids[0]))

    # Generate with steering
    hook_fn = _make_steering_hook(steering_vector, options.coefficient, device)
    steered_ids = _generate_tokens(
        model, tokenizer, ids, options.max_new_tokens,
        hook_layer=target_layer, hook_fn=hook_fn,
    )
    steered_text = _decode_generated(tokenizer, steered_ids, len(ids[0]))

    result: dict[str, Any] = {
        "input_text": options.input_text,
        "original_text": original_text,
        "steered_text": steered_text,
        "coefficient": options.coefficient,
        "layer_name": target_layer,
        "max_new_tokens": options.max_new_tokens,
    }

    out_dir = Path(options.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "steer_apply.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def _make_steering_hook(
    vector: Tensor, coefficient: float, device: Any,
) -> Callable[..., Any]:
    """Create a forward hook that adds the steering vector to activations."""
    vec = vector.to(device)

    def hook(_module: Any, _input: Any, output: Any) -> Any:
        target = output[0] if isinstance(output, tuple) else output
        scaled = coefficient * vec.to(dtype=target.dtype)
        if isinstance(output, tuple):
            return (output[0] + scaled,) + output[1:]
        return output + scaled

    return hook


def _generate_tokens(
    model: Any, tokenizer: Any, input_ids: Tensor,
    max_new_tokens: int,
    hook_layer: str = "", hook_fn: Any = None,
) -> Tensor:
    """Simple greedy generation loop."""
    from serve.tokenization import collect_stop_token_ids

    current_ids = input_ids.clone()
    hook_handle = None
    stop_ids = collect_stop_token_ids(tokenizer)

    if hook_layer and hook_fn:
        module = _get_module_by_name(model, hook_layer)
        hook_handle = module.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(current_ids)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs
                next_token = logits[0, -1].argmax(dim=-1, keepdim=True)
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
                if int(next_token) in stop_ids:
                    break
    finally:
        if hook_handle:
            hook_handle.remove()

    return current_ids


def _decode_generated(tokenizer: Any, ids: Tensor, prompt_len: int) -> str:
    """Decode only the generated portion of the output."""
    generated_ids = ids[0, prompt_len:].tolist()
    tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    # BPE tokens use \u0120 (Ġ) for leading space, \u010a (Ċ) for newline
    text = "".join(tokens)
    text = text.replace("\u0120", " ").replace("\u010a", "\n")
    return text


def _get_module_by_name(model: Any, name: str) -> Any:
    """Resolve a dotted module name to a module."""
    parts = name.split(".")
    current = model
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _load_model_and_tokenizer(options: SteerApplyOptions) -> tuple[Any, Any]:
    from serve.interp_model_loader import load_interp_model
    return load_interp_model(options.base_model or options.model_path)
