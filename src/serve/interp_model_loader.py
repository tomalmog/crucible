"""Shared model + tokenizer loading for interpretability analyses.

Single entry point for all interp tools (logit lens, activation PCA,
activation patching). Handles HuggingFace models, Crucible .pt checkpoints,
and any tokenizer type. Returns a uniform InterpTokenizer so runners never
need to care about model origin.
"""

from __future__ import annotations

from typing import Any


def load_interp_model(model_path: str) -> tuple[Any, "InterpTokenizer"]:
    """Load model + tokenizer for interpretability analysis.

    Supports:
    - HuggingFace model IDs (e.g. ``gpt2``, ``meta-llama/Llama-2-7b``)
    - HuggingFace model directories (with ``config.json``)
    - Crucible ``.pt`` checkpoints (with ``vocab.json`` or ``tokenizer.json``)

    Returns ``(model, tokenizer)`` where *tokenizer* is always an
    ``InterpTokenizer`` with a consistent API regardless of model origin.
    """
    from serve.hf_model_loader import is_huggingface_model_id

    if is_huggingface_model_id(model_path):
        return _load_hf(model_path)
    # Check if this is a merged HF model (e.g. from LoRA/QLoRA on an HF base).
    hf_base = _detect_hf_base_model(model_path)
    if hf_base:
        return _load_hf_with_weights(hf_base, model_path)
    return _load_crucible(model_path)


# ---------- HuggingFace path ----------


def _load_hf(model_path: str) -> tuple[Any, "InterpTokenizer"]:
    """Load a HuggingFace model + tokenizer."""
    from serve.hf_model_loader import (
        load_huggingface_model,
        load_huggingface_tokenizer,
    )

    model = load_huggingface_model(model_path)
    hf_tok = load_huggingface_tokenizer(model_path)
    return model, InterpTokenizer.from_hf(hf_tok)


def _load_hf_with_weights(
    hf_model_id: str, weights_path: str,
) -> tuple[Any, "InterpTokenizer"]:
    """Load HF architecture with custom weights from a .pt checkpoint.

    Used for merged LoRA/QLoRA models where the .pt file contains
    HF-style keys but the architecture comes from a HF base model.
    """
    from serve.hf_model_loader import (
        load_huggingface_model,
        load_huggingface_tokenizer,
    )

    model = load_huggingface_model(hf_model_id, weights_path)
    model.eval()
    hf_tok = load_huggingface_tokenizer(hf_model_id)
    return model, InterpTokenizer.from_hf(hf_tok)


def _detect_hf_base_model(model_path: str) -> str | None:
    """Check if model_path has a training_config with a HF base_model_path."""
    from serve.hf_model_loader import is_huggingface_model_id
    from serve.training_metadata import load_training_config

    config = load_training_config(model_path)
    if not config:
        return None
    for key in ("base_model_path", "initial_weights_path"):
        value = str(config.get(key, "") or "")
        if not value:
            continue
        if is_huggingface_model_id(value):
            return value
        resolved = _resolve_remote_hf_id(value)
        if resolved:
            return resolved
    return None


def _resolve_remote_hf_id(remote_path: str) -> str | None:
    """Try to recover a HuggingFace model ID from a remote cluster path."""
    from pathlib import Path as _Path
    from serve.hf_model_loader import is_huggingface_model_id

    basename = _Path(remote_path).name
    if not basename:
        return None
    idx = basename.find("_")
    if idx > 0:
        candidate = basename[:idx] + "/" + basename[idx + 1:]
        if is_huggingface_model_id(candidate):
            return candidate
    if is_huggingface_model_id(basename):
        return basename
    return None


# ---------- Crucible path ----------


def _load_crucible(model_path: str) -> tuple[Any, "InterpTokenizer"]:
    """Load a Crucible .pt checkpoint + tokenizer."""
    import torch

    from core.chat_types import ChatOptions
    from serve.architecture_loader import load_training_model
    from serve.chat_option_resolver import (
        resolve_chat_model_vocab_size,
        resolve_chat_training_options,
    )
    from serve.device_selection import resolve_execution_device
    from serve.model_weights import load_initial_weights, read_model_state_dict
    from serve.training_metadata import load_tokenizer

    device = resolve_execution_device(torch)
    model_state = read_model_state_dict(torch, model_path, device)

    chat_opts = ChatOptions(model_path=model_path, prompt="")
    training_options = resolve_chat_training_options(chat_opts, model_state)

    tokenizer = load_tokenizer(model_path)
    if tokenizer is None:
        from core.errors import CrucibleServeError

        raise CrucibleServeError(
            f"No tokenizer found beside model at {model_path}. "
            "Ensure vocab.json exists in the model directory."
        )

    vocab_size = resolve_chat_model_vocab_size(
        tokenizer.vocabulary, model_state, training_options,
    )
    model = load_training_model(torch, training_options, vocab_size)
    model = model.to(device)
    load_initial_weights(torch, model, model_path, device)
    model.eval()

    return model, InterpTokenizer.from_crucible(tokenizer)


# ---------- Unified tokenizer ----------


class InterpTokenizer:
    """Uniform tokenizer interface used by all interp runners.

    Wraps any underlying tokenizer (HF AutoTokenizer, Crucible
    VocabularyTokenizer, Crucible HuggingFaceTokenizer) so that runners
    only ever call these methods:

    - ``encode(text, return_tensors="pt") -> Tensor``
    - ``convert_ids_to_tokens(ids) -> list[str]``
    """

    def __init__(
        self,
        encode_fn: Any,
        id_to_token: dict[int, str],
    ) -> None:
        self._encode_fn = encode_fn
        self._id_to_token = id_to_token

    # -- Construction helpers --

    @staticmethod
    def from_hf(hf_tokenizer: Any) -> "InterpTokenizer":
        """Wrap a ``transformers.AutoTokenizer``."""
        vocab: dict[str, int] = dict(hf_tokenizer.get_vocab())
        reverse = {v: k for k, v in vocab.items()}

        def _encode(
            text: str,
            return_tensors: str | None = None,
            truncation: bool = True,
            max_length: int = 512,
        ) -> Any:
            return hf_tokenizer.encode(
                text,
                return_tensors=return_tensors,
                truncation=truncation,
                max_length=max_length,
            )

        return InterpTokenizer(_encode, reverse)

    @staticmethod
    def from_crucible(chat_tokenizer: Any) -> "InterpTokenizer":
        """Wrap a Crucible ``ChatTokenizer`` (VocabularyTokenizer or
        HuggingFaceTokenizer from ``huggingface_tokenizer.py``).
        """
        vocab: dict[str, int] = dict(chat_tokenizer.vocabulary)
        reverse = {v: k for k, v in vocab.items()}

        def _encode(
            text: str,
            return_tensors: str | None = None,
            truncation: bool = True,
            max_length: int = 512,
        ) -> Any:
            ids = chat_tokenizer.encode(text, max_length)
            if return_tensors == "pt":
                import torch

                return torch.tensor([ids], dtype=torch.long)
            return ids

        return InterpTokenizer(_encode, reverse)

    # -- Public API (used by all runners) --

    def encode(
        self,
        text: str,
        return_tensors: str | None = None,
        truncation: bool = True,
        max_length: int = 512,
    ) -> Any:
        """Encode text to token IDs or tensor."""
        return self._encode_fn(
            text,
            return_tensors=return_tensors,
            truncation=truncation,
            max_length=max_length,
        )

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """Map token IDs back to string tokens."""
        return [self._id_to_token.get(i, f"[{i}]") for i in ids]
