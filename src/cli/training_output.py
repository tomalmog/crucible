"""Shared training result output and auto-registration.

Every CLI training command should call ``print_and_register`` after
training completes.  This prints the result fields for Tauri/UI
consumption AND auto-registers the model in the model registry so
it appears on the Models page immediately.
"""

from __future__ import annotations

from pathlib import Path

from core.types import TrainingRunResult
from store.dataset_sdk import CrucibleClient


def print_and_register(
    client: CrucibleClient,
    result: TrainingRunResult,
    model_name: str | None = None,
) -> None:
    """Print training result and auto-register the model.

    Args:
        client: SDK client (provides data_root for registry).
        result: Completed training run result.
        model_name: Optional model name for registry. If not provided,
            derives a name from the output directory.
    """
    # Print result fields (consumed by Tauri task store)
    print(f"model_path={result.model_path}")
    print(f"history_path={result.history_path}")
    print(f"plot_path={result.plot_path or '-'}")
    print(f"epochs_completed={result.epochs_completed}")
    print(f"checkpoint_dir={result.checkpoint_dir or '-'}")
    print(f"best_checkpoint_path={result.best_checkpoint_path or '-'}")
    print(f"run_id={result.run_id or '-'}")
    print(f"artifact_contract_path={result.artifact_contract_path or '-'}")

    # Auto-register in model registry
    if result.model_path:
        _auto_register(client, result, model_name)


def _auto_register(
    client: CrucibleClient,
    result: TrainingRunResult,
    model_name: str | None,
) -> None:
    """Register the trained model so it appears in the UI."""
    from store.model_registry import ModelRegistry

    if not model_name:
        model_name = _derive_model_name(result.model_path)

    try:
        registry = ModelRegistry(client._config.data_root)
        registry.register_model(
            model_name=model_name,
            model_path=result.model_path,
            run_id=result.run_id,
        )
    except Exception:
        # Registration failure should not break the training command
        pass


def _derive_model_name(model_path: str) -> str:
    """Derive a model name from the output path."""
    p = Path(model_path)
    # Use the parent directory name (e.g. "sft-output" from ".../sft-output/model.pt")
    parent = p.parent.name
    if parent in ("output", "hf_model"):
        parent = p.parent.parent.name
    return parent or "trained-model"
