"""GGUF model export runner.

Exports a PyTorch model (HuggingFace or Crucible .pt) to GGUF format
with configurable quantization type.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.errors import CrucibleDependencyError
from core.gguf_export_types import GgufExportOptions
from serve.export_helpers import (
    classify_model,
    copy_tokenizer,
    load_state_dict_for_export,
    model_basename,
)

_QUANTIZED_TYPES = frozenset({"Q8_0", "Q4_0", "Q4_K_M", "Q5_K_M"})


def run_gguf_export(options: GgufExportOptions) -> dict[str, Any]:
    """Export a model to GGUF format.

    Expects model_path to be pre-resolved (local path or HF model ID).
    """
    try:
        import gguf  # noqa: F811
    except ImportError as error:
        raise CrucibleDependencyError(
            "GGUF export requires the gguf package. "
            "Install with: pip install 'crucible[gguf]'"
        ) from error

    if options.quant_type in _QUANTIZED_TYPES:
        raise CrucibleDependencyError(
            f"Quantized export ({options.quant_type}) requires llama.cpp. "
            "Use F32 or F16 for direct export, or install llama-cpp-python "
            "and use its quantize utility after F16 export."
        )

    output_dir = Path(options.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    is_hf, hf_model_id = classify_model(options.model_path)
    name = model_basename(options.model_path)
    output_path = output_dir / f"{name}-{options.quant_type}.gguf"

    if is_hf:
        num_tensors = _export_hf(hf_model_id, output_path, options, gguf)
    else:
        num_tensors = _export_crucible(
            options.model_path, output_path, options, gguf,
        )

    tok_source = hf_model_id if is_hf else options.model_path
    tokenizer_copied = copy_tokenizer(tok_source, output_dir, is_hf)
    file_size_mb = round(output_path.stat().st_size / (1024 * 1024), 2)

    result: dict[str, Any] = {
        "output_path": str(output_path),
        "file_size_mb": file_size_mb,
        "quant_type": options.quant_type,
        "num_tensors": num_tensors,
        "tokenizer_copied": tokenizer_copied,
    }
    print(json.dumps(result, indent=2))
    return result


def _export_hf(
    model_id: str,
    output_path: Path,
    options: GgufExportOptions,
    gguf_mod: Any,
) -> int:
    """Export HuggingFace model to GGUF."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    state = dict(model.state_dict())

    return _write_gguf(output_path, state, config, options, gguf_mod)


def _export_crucible(
    model_path: str,
    output_path: Path,
    options: GgufExportOptions,
    gguf_mod: Any,
) -> int:
    """Export Crucible .pt model to GGUF."""
    state = load_state_dict_for_export(model_path)

    config = _infer_config_from_state(state)
    return _write_gguf(output_path, state, config, options, gguf_mod)


def _infer_config_from_state(state: dict[str, Any]) -> _SimpleConfig:
    """Infer architecture metadata from a state dict."""
    hidden_dim = 0
    vocab_size = 0
    num_layers = 0

    for key, tensor in state.items():
        if "embed" in key and tensor.dim() == 2:
            vocab_size, hidden_dim = tensor.shape
        if "layer" in key or "block" in key:
            parts = key.split(".")
            for p in parts:
                if p.isdigit():
                    num_layers = max(num_layers, int(p) + 1)

    return _SimpleConfig(
        hidden_size=hidden_dim or 768,
        num_hidden_layers=num_layers or 12,
        num_attention_heads=max(hidden_dim // 64, 1) if hidden_dim else 12,
        vocab_size=vocab_size or 50257,
    )


class _SimpleConfig:
    """Minimal config for Crucible models lacking HF config.json."""

    def __init__(
        self, hidden_size: int, num_hidden_layers: int,
        num_attention_heads: int, vocab_size: int,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size


def _write_gguf(
    output_path: Path,
    state: dict[str, Any],
    config: Any,
    options: GgufExportOptions,
    gguf_mod: Any,
) -> int:
    """Write model tensors to a GGUF file."""
    import torch

    writer = gguf_mod.GGUFWriter(str(output_path), "llama")

    # Architecture metadata
    writer.add_embedding_length(getattr(config, "hidden_size", 768))
    writer.add_block_count(getattr(config, "num_hidden_layers", 12))
    writer.add_head_count(getattr(config, "num_attention_heads", 12))

    num_tensors = 0
    use_f16 = options.quant_type == "F16"

    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        t = tensor.detach().cpu()
        if use_f16 and t.is_floating_point() and t.dtype != torch.float16:
            t = t.to(torch.float16)
        writer.add_tensor(name, t.numpy())
        num_tensors += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return num_tensors
