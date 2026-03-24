"""SafeTensors model export runner.

Exports a PyTorch model (HuggingFace or Crucible .pt) to SafeTensors format
with tokenizer copying.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.errors import CrucibleDependencyError
from core.safetensors_export_types import SafeTensorsExportOptions
from serve.export_helpers import (
    classify_model,
    copy_tokenizer,
    load_state_dict_for_export,
    model_basename,
)


def run_safetensors_export(options: SafeTensorsExportOptions) -> dict[str, Any]:
    """Export a model to SafeTensors format.

    Expects model_path to be pre-resolved (local path or HF model ID).
    """
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise CrucibleDependencyError(
            "SafeTensors export requires the safetensors package. "
            "Install with: pip install safetensors"
        ) from error

    output_dir = Path(options.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    is_hf, hf_model_id = classify_model(options.model_path)

    if is_hf:
        output_path, num_tensors = _export_hf(hf_model_id, output_dir)
    else:
        output_path, num_tensors = _export_crucible(
            options.model_path, output_dir, save_file,
        )

    tok_source = hf_model_id if is_hf else options.model_path
    tokenizer_copied = copy_tokenizer(tok_source, output_dir, is_hf)
    file_size_mb = round(output_path.stat().st_size / (1024 * 1024), 2)

    result: dict[str, Any] = {
        "output_path": str(output_path),
        "file_size_mb": file_size_mb,
        "num_tensors": num_tensors,
        "tokenizer_copied": tokenizer_copied,
    }
    print(json.dumps(result, indent=2))
    return result


def _export_hf(model_id: str, output_dir: Path) -> tuple[Path, int]:
    """Export HuggingFace model to SafeTensors via save_pretrained."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    num_tensors = len(dict(model.state_dict()))

    # Find the saved safetensors file
    st_files = list(output_dir.glob("*.safetensors"))
    if st_files:
        return st_files[0], num_tensors
    return output_dir / "model.safetensors", num_tensors


def _export_crucible(
    model_path: str, output_dir: Path, save_file: Any,
) -> tuple[Path, int]:
    """Export a Crucible .pt model to SafeTensors."""
    state = load_state_dict_for_export(model_path)
    name = model_basename(model_path)
    output_path = output_dir / f"{name}.safetensors"
    save_file(state, str(output_path))
    return output_path, len(state)
