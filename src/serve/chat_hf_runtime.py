"""HuggingFace chat runtime policies."""

from __future__ import annotations

import contextlib
import io
import os
from collections.abc import Iterator
from typing import Any

from serve.device_selection import resolve_execution_device


def resolve_hf_inference_device(torch_module: Any) -> Any:
    """Use CUDA when available, but avoid Apple MPS for HF chat generation."""
    device = resolve_execution_device(torch_module)
    if str(getattr(device, "type", device)) == "mps":
        return torch_module.device("cpu")
    return device


@contextlib.contextmanager
def quiet_hf_loading() -> Iterator[None]:
    """Suppress HuggingFace progress output while preserving token streaming."""
    old_hub = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    old_transformers = os.environ.get("TRANSFORMERS_NO_ADVISORY_WARNINGS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        _restore_env("HF_HUB_DISABLE_PROGRESS_BARS", old_hub)
        _restore_env("TRANSFORMERS_NO_ADVISORY_WARNINGS", old_transformers)


def _restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
        return
    os.environ[key] = value
