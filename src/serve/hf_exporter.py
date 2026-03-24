"""HuggingFace model export runner.

Exports a model to a complete HuggingFace-compatible directory with
model.safetensors, config.json, and tokenizer files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.errors import CrucibleDependencyError
from core.hf_export_types import HfExportOptions
from serve.export_helpers import (
    classify_model,
    copy_crucible_tokenizer,
    load_state_dict_for_export,
)


def run_hf_export(options: HfExportOptions) -> dict[str, Any]:
    """Export a model to HuggingFace-compatible format.

    Expects model_path to be pre-resolved (local path or HF model ID).
    """
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise CrucibleDependencyError(
            "HuggingFace export requires the safetensors package. "
            "Install with: pip install safetensors"
        ) from error

    output_dir = Path(options.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    is_hf, hf_model_id = classify_model(options.model_path)

    if is_hf:
        output_path, num_tensors = _export_hf(hf_model_id, output_dir)
        config_generated = False
        tokenizer_copied = True
    else:
        output_path, num_tensors, config_generated = _export_crucible(
            options.model_path, output_dir, save_file,
        )
        tokenizer_copied = copy_crucible_tokenizer(
            options.model_path, output_dir,
        )

    file_size_mb = round(output_path.stat().st_size / (1024 * 1024), 2)

    result: dict[str, Any] = {
        "output_path": str(output_dir),
        "file_size_mb": file_size_mb,
        "num_tensors": num_tensors,
        "config_generated": config_generated,
        "tokenizer_copied": tokenizer_copied,
    }
    print(json.dumps(result, indent=2))
    return result


def _export_hf(model_id: str, output_dir: Path) -> tuple[Path, int]:
    """Export HuggingFace model via save_pretrained (complete output)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    num_tensors = len(dict(model.state_dict()))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(str(output_dir))

    st_files = list(output_dir.glob("*.safetensors"))
    output_path = st_files[0] if st_files else output_dir / "model.safetensors"
    return output_path, num_tensors


def _export_crucible(
    model_path: str, output_dir: Path, save_file: Any,
) -> tuple[Path, int, bool]:
    """Export a Crucible .pt model with generated config.json."""
    state = load_state_dict_for_export(model_path)
    output_path = output_dir / "model.safetensors"
    save_file(state, str(output_path))
    config_generated = _generate_config_json(model_path, output_dir, state)
    return output_path, len(state), config_generated


def _generate_config_json(
    model_path: str, output_dir: Path, state_dict: dict[str, Any],
) -> bool:
    """Generate a config.json from training_config.json and state dict."""
    from serve.training_metadata import load_training_config

    config: dict[str, Any] = {
        "model_type": "crucible",
        "architectures": ["CrucibleTransformerLM"],
        "hidden_act": "gelu",
        "layer_norm_epsilon": 1e-5,
    }

    # Infer vocab_size from embedding weight shape
    for key in ("embedding.weight", "embed_tokens.weight", "wte.weight"):
        if key in state_dict:
            config["vocab_size"] = state_dict[key].shape[0]
            break

    training_config = load_training_config(model_path)
    if training_config:
        field_map = {
            "hidden_dim": "hidden_size",
            "num_layers": "num_hidden_layers",
            "attention_heads": "num_attention_heads",
            "mlp_hidden_dim": "intermediate_size",
            "max_token_length": "max_position_embeddings",
            "mlp_layers": "num_output_layers",
            "position_embedding_type": "position_embedding_type",
        }
        for src, dst in field_map.items():
            val = training_config.get(src)
            if val is not None:
                config[dst] = val

        dropout = training_config.get("dropout")
        if dropout is not None:
            config["hidden_dropout_prob"] = dropout
            config["attention_probs_dropout_prob"] = dropout

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return True
