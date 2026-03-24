"""ONNX model export runner.

Exports a PyTorch model (HuggingFace, Crucible .pt, or LoRA) to ONNX format
with tokenizer copying and verification via ONNX Runtime.

HuggingFace models are exported via the ``optimum`` library which handles
the complexities of modern transformers architectures. Crucible .pt models
use the standard ``torch.onnx.export`` path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.errors import CrucibleDependencyError, CrucibleExportError
from core.onnx_export_types import OnnxExportOptions
from serve.export_helpers import (
    classify_model,
    copy_tokenizer,
    load_crucible_model,
    model_basename,
)


def run_onnx_export(options: OnnxExportOptions) -> dict[str, Any]:
    """Export a model to ONNX format.

    Expects model_path to be pre-resolved (local path or HF model ID).
    """
    _onnx_mod, ort_mod = _import_onnx_deps()

    output_dir = Path(options.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    name = model_basename(options.model_path)
    onnx_path = output_dir / f"{name}.onnx"

    is_hf, hf_model_id = classify_model(options.model_path)

    if is_hf:
        input_names = _export_hf(
            hf_model_id, output_dir, onnx_path,
        )
    else:
        input_names = _export_crucible(options, onnx_path)

    tok_source = hf_model_id if is_hf else options.model_path
    tokenizer_copied = copy_tokenizer(tok_source, output_dir, is_hf)
    verification = _verify_export(ort_mod, str(onnx_path), input_names)
    file_size_mb = round(onnx_path.stat().st_size / (1024 * 1024), 2)

    result = {
        "onnx_path": str(onnx_path),
        "file_size_mb": file_size_mb,
        "opset_version": options.opset_version,
        "verification": verification,
        "input_names": input_names,
        "output_names": ["logits"],
        "tokenizer_copied": tokenizer_copied,
    }
    print(json.dumps(result, indent=2))
    return result


# -- HuggingFace export (via optimum) ------------------------------------


def _export_hf(model_path: str, output_dir: Path, onnx_path: Path) -> list[str]:
    """Export HuggingFace model via optimum's ONNX exporter."""
    try:
        from optimum.exporters.onnx import main_export
    except ImportError as error:
        raise CrucibleDependencyError(
            "Exporting HuggingFace models to ONNX requires optimum. "
            "Install with: pip install 'optimum[onnxruntime]'"
        ) from error

    main_export(
        model_name_or_path=model_path,
        output=str(output_dir),
        task="text-generation",
        no_post_process=True,
    )

    # optimum creates model.onnx -- rename to our desired name
    optimum_output = output_dir / "model.onnx"
    if optimum_output.exists() and optimum_output != onnx_path:
        optimum_output.rename(onnx_path)

    # Determine input names from the exported model
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    return [inp.name for inp in session.get_inputs()]


# -- Crucible export (torch.onnx.export) ----------------------------------


def _export_crucible(options: OnnxExportOptions, onnx_path: Path) -> list[str]:
    """Export a Crucible .pt model via torch.onnx.export."""
    import torch

    model, _tokenizer = load_crucible_model(options.model_path)
    model.eval()
    model = model.to("cpu")

    input_ids = torch.ones(1, 8, dtype=torch.long)
    input_names = ["input_ids"]
    output_names = ["logits"]

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    }

    torch.onnx.export(
        model,
        (input_ids,),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=options.opset_version,
    )
    return input_names


# -- Shared helpers -------------------------------------------------------


def _import_onnx_deps() -> tuple[Any, Any]:
    """Import onnx and onnxruntime."""
    try:
        import onnx
    except ImportError as error:
        raise CrucibleDependencyError(
            "ONNX export requires the onnx package. "
            "Install with: pip install -e .[onnx]"
        ) from error
    try:
        import onnxruntime
    except ImportError as error:
        raise CrucibleDependencyError(
            "ONNX export requires onnxruntime. "
            "Install with: pip install -e .[onnx]"
        ) from error
    return onnx, onnxruntime


def _verify_export(ort_mod: Any, onnx_path: str, input_names: list[str]) -> str:
    """Load the exported ONNX model and run a dummy forward pass."""
    import numpy as np

    try:
        providers = list(ort_mod.get_available_providers())
        session = ort_mod.InferenceSession(onnx_path, providers=providers)
        feed: dict[str, Any] = {}
        for name in input_names:
            feed[name] = np.ones((1, 8), dtype=np.int64)
        outputs = session.run(None, feed)
        if outputs and len(outputs[0].shape) == 3:
            return "passed"
        return "failed — unexpected output shape"
    except Exception as error:
        return f"failed — {error}"
