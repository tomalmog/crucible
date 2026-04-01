"""Shared helpers for model export (ONNX, SafeTensors, GGUF).

Provides validation, model classification, tokenizer copying, and
path utilities used by all export runners.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from core.errors import CrucibleExportError


def resolve_model_path(model_path: str, data_root: Path | None = None) -> str:
    """Resolve a model path for local export.

    If the path exists locally, returns it as-is.  If the path is absolute
    and doesn't exist locally (i.e. a remote cluster path), attempts:
    1. Registry lookup — find a model entry whose remote_path matches and
       return its local model_path (if it has one).
    2. HF ID recovery — extract the basename and reverse the
       sanitize_remote_name transformation to recover an org/model ID.
    3. Fail with a clear error asking the user to pull the model first.
    """
    path = Path(model_path).expanduser()
    if not path.is_absolute() or path.exists():
        return model_path

    # For non-.pt remote paths, try HF ID recovery first (no pull needed)
    basename = path.name
    is_pt = basename.endswith((".pt", ".pth", ".bin"))
    if basename and not is_pt:
        candidate = _unsanitize_hf_id(basename)
        if candidate:
            return candidate

    # For .pt files, try registry lookup (may auto-pull)
    if data_root is not None:
        local = _find_local_path_for_remote(model_path, data_root)
        if local:
            return local

    raise CrucibleExportError(
        f"Model path does not exist: {model_path}\n"
        "Export requires model files on disk. "
        "If this is a remote model, pull it first with: "
        "crucible model pull --name <model-name>"
    )


def _find_local_path_for_remote(remote_path: str, data_root: Path) -> str | None:
    """Search the model registry for a local path matching a remote path.

    If the model is remote-only, automatically pulls it to local storage
    so the export can proceed.
    """
    try:
        from store.model_registry import ModelRegistry

        registry = ModelRegistry(data_root)
        for entry in registry.list_models():
            if entry.remote_path != remote_path:
                continue
            # Already has a local copy
            if entry.model_path:
                local = Path(entry.model_path).expanduser()
                if local.exists():
                    return entry.model_path
            # Remote-only — auto-pull
            return _auto_pull_model(data_root, entry)
    except CrucibleExportError:
        raise
    except Exception:
        pass
    return None


def _auto_pull_model(data_root: Path, entry: Any) -> str | None:
    """Pull a remote model to local storage and return the local path."""
    print(f"Pulling remote model '{entry.model_name}' for local export...")
    try:
        pulled = False
        if entry.run_id:
            try:
                from serve.remote_model_puller import pull_remote_model

                pull_remote_model(data_root, entry.run_id, entry.model_name)
                pulled = True
            except Exception:
                pass  # Fall through to direct pull
        if not pulled:
            from serve.remote_model_puller import pull_remote_model_direct

            pull_remote_model_direct(
                data_root, entry.model_name,
                entry.remote_host, entry.remote_path,
            )
        # Re-read the entry to get the local path set by mark_model_pulled
        from store.model_registry import ModelRegistry

        registry = ModelRegistry(data_root)
        updated = registry.get_model(entry.model_name)
        if updated.model_path and Path(updated.model_path).expanduser().exists():
            return updated.model_path
    except Exception as exc:
        raise CrucibleExportError(
            f"Failed to pull remote model '{entry.model_name}': {exc}\n"
            "Pull manually with: crucible model pull "
            f"--name {entry.model_name}"
        ) from exc
    return None


def _unsanitize_hf_id(basename: str) -> str | None:
    """Try to recover a HuggingFace org/model ID from a sanitized basename.

    sanitize_remote_name turns ``org/model`` into ``org_model``.  We try
    each underscore position as a potential ``/`` to recover org/model
    format first, then fall back to the basename as-is for simple IDs
    like "gpt2".
    """
    from serve.hf_model_loader import is_huggingface_model_id

    # Try replacing each underscore with a slash first (prefer org/model form)
    # e.g. "openai-community_gpt2" → "openai-community/gpt2"
    for i, ch in enumerate(basename):
        if ch == "_":
            candidate = basename[:i] + "/" + basename[i + 1:]
            if is_huggingface_model_id(candidate):
                return candidate

    # Fall back to basename as-is (e.g. "gpt2")
    if is_huggingface_model_id(basename):
        return basename

    return None


def classify_model(model_path: str) -> tuple[bool, str]:
    """Determine if path is HuggingFace and return (is_hf, hf_model_id).

    For HF models, hf_model_id is the path/ID to pass to from_pretrained.
    For LoRA/QLoRA .pt files trained on an HF base, detects the base
    model and returns it as the HF ID.
    For Crucible .pt models, returns (False, "").
    """
    from serve.hf_model_loader import is_huggingface_model_id

    if is_huggingface_model_id(model_path):
        return True, model_path

    path = Path(model_path).expanduser()
    if path.suffix in (".pt", ".pth") and path.exists():
        hf_base = detect_hf_base_model(str(path))
        if hf_base:
            return True, hf_base

    return False, ""


def detect_hf_base_model(model_path: str) -> str | None:
    """Check training_config.json for an HF base model path."""
    from serve.model_type_detection import detect_hf_base_model as _detect
    return _detect(model_path)


def copy_tokenizer(model_path: str, output_dir: Path, is_hf: bool) -> bool:
    """Copy tokenizer files to the output directory."""
    if is_hf:
        return copy_hf_tokenizer(model_path, output_dir)
    return copy_crucible_tokenizer(model_path, output_dir)


def copy_hf_tokenizer(model_path: str, output_dir: Path) -> bool:
    """Save HuggingFace tokenizer to output directory."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.save_pretrained(str(output_dir))
        return True
    except Exception:
        return False


def copy_crucible_tokenizer(model_path: str, output_dir: Path) -> bool:
    """Copy Crucible tokenizer files to the output directory."""
    from serve.training_metadata import _artifact_dir

    artifact_dir = _artifact_dir(model_path)
    copied = False
    for name in ("tokenizer_vocab.json", "vocab.json", "tokenizer.json"):
        src = artifact_dir / name
        if src.exists():
            shutil.copy2(str(src), str(output_dir / name))
            copied = True
    return copied


def model_basename(model_path: str) -> str:
    """Extract a clean basename from a model path or ID."""
    path = Path(model_path)
    if path.suffix in (".pt", ".pth", ".bin", ".safetensors"):
        return path.stem
    name = path.name or model_path
    return name.replace("/", "_")


def load_crucible_model(model_path: str) -> tuple[Any, Any]:
    """Load a Crucible model and tokenizer via the interp model loader."""
    from serve.interp_model_loader import load_interp_model

    try:
        return load_interp_model(model_path)
    except Exception as error:
        raise CrucibleExportError(
            f"Failed to load model from '{model_path}': {error}"
        ) from error


def load_state_dict_for_export(model_path: str) -> dict[str, Any]:
    """Load a state dict from a .pt file for export.

    For LoRA/QLoRA .pt files (detected via training_config.json with an
    HF base model), loads the HF base model, applies the LoRA state dict,
    and returns the merged weights so the export contains the full
    fine-tuned model rather than the un-trained base.

    For plain .pt files, loads the raw tensor dict directly.
    """
    import torch

    path = Path(model_path).expanduser()
    if not path.exists():
        raise CrucibleExportError(f"Model file not found: {model_path}")

    # Check if this is a LoRA .pt trained on an HF base model
    hf_base = detect_hf_base_model(str(path))
    if hf_base:
        return _load_lora_merged_state_dict(str(path), hf_base, torch)

    try:
        data = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(data, dict) and "state_dict" in data:
            return dict(data["state_dict"])
        if isinstance(data, dict):
            return dict(data)
        raise CrucibleExportError(
            f"Unexpected format in '{model_path}': expected dict, "
            f"got {type(data).__name__}"
        )
    except CrucibleExportError:
        raise
    except Exception as error:
        raise CrucibleExportError(
            f"Failed to load state dict from '{model_path}': {error}"
        ) from error


def _load_lora_merged_state_dict(
    model_path: str, hf_base_id: str, torch_module: Any,
) -> dict[str, Any]:
    """Load HF base model, apply LoRA .pt weights, return merged state dict."""
    try:
        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(
            hf_base_id, torch_dtype=torch_module.float32,
        )
        lora_state = torch_module.load(
            model_path, map_location="cpu", weights_only=True,
        )
        if isinstance(lora_state, dict) and "state_dict" in lora_state:
            lora_state = lora_state["state_dict"]

        # Apply LoRA weights on top of the base model
        base_state = base_model.state_dict()
        base_state.update(lora_state)
        base_model.load_state_dict(base_state, strict=False)
        return dict(base_model.state_dict())
    except CrucibleExportError:
        raise
    except Exception as error:
        raise CrucibleExportError(
            f"Failed to merge LoRA weights from '{model_path}' "
            f"with base model '{hf_base_id}': {error}"
        ) from error
