"""Training checkpoint persistence helpers.

This module saves epoch checkpoints, tracks best-model snapshots, applies
retention policies, and restores model/optimizer state for resume workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.constants import (
    DEFAULT_TRAIN_BEST_CHECKPOINT_FILE_NAME,
    DEFAULT_TRAIN_CHECKPOINT_DIR_NAME,
)
from core.errors import CrucibleServeError


@dataclass(frozen=True)
class CheckpointResumeState:
    """Resume state loaded from a saved training checkpoint."""

    next_epoch: int
    global_step: int
    best_validation_loss: float | None
    checkpoint_path: Path


def ensure_checkpoint_dir(output_dir: Path) -> Path:
    """Create and return checkpoint directory under training output."""
    checkpoint_dir = output_dir / DEFAULT_TRAIN_CHECKPOINT_DIR_NAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_epoch_checkpoint(
    checkpoint_dir: Path,
    torch_module: Any,
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    epoch: int,
    global_step: int,
    best_validation_loss: float | None,
) -> Path:
    """Save periodic epoch checkpoint and return file path."""
    checkpoint_path = checkpoint_dir / _epoch_checkpoint_file_name(epoch)
    payload = _build_checkpoint_payload(
        model,
        optimizer,
        scheduler,
        epoch,
        global_step,
        best_validation_loss,
    )
    _save_checkpoint_payload(torch_module, checkpoint_path, payload)
    return checkpoint_path


def save_best_checkpoint(
    checkpoint_dir: Path,
    torch_module: Any,
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
) -> Path:
    """Save best-model checkpoint and return path."""
    checkpoint_path = checkpoint_dir / DEFAULT_TRAIN_BEST_CHECKPOINT_FILE_NAME
    payload = _build_checkpoint_payload(
        model,
        optimizer,
        scheduler,
        epoch,
        global_step,
        best_validation_loss,
    )
    _save_checkpoint_payload(torch_module, checkpoint_path, payload)
    return checkpoint_path


def prune_epoch_checkpoints(checkpoint_dir: Path, max_files: int | None) -> None:
    """Remove oldest epoch checkpoints to satisfy retention policy."""
    if max_files is None:
        return
    epoch_files = _list_epoch_checkpoints(checkpoint_dir)
    if len(epoch_files) <= max_files:
        return
    for stale_path in epoch_files[: len(epoch_files) - max_files]:
        try:
            stale_path.unlink()
        except OSError as error:
            raise CrucibleServeError(
                f"Failed to remove old checkpoint {stale_path}: {error}. "
                "Check output directory permissions and retry."
            ) from error


def load_resume_checkpoint(
    checkpoint_path: str,
    torch_module: Any,
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    device: Any,
) -> CheckpointResumeState:
    """Load checkpoint payload and apply model/optimizer state."""
    resolved_path = Path(checkpoint_path).expanduser().resolve()
    payload = _read_checkpoint_payload(resolved_path, torch_module, device)
    model_state = _read_mapping(payload, "model_state_dict", resolved_path)
    optimizer_state = _read_mapping(payload, "optimizer_state_dict", resolved_path)
    epoch = _read_epoch(payload, resolved_path)
    global_step = _read_global_step(payload, resolved_path)
    best_validation_loss = _read_optional_float(payload, "best_validation_loss", resolved_path)
    scheduler_state = _read_optional_mapping(payload, "scheduler_state_dict", resolved_path)
    from serve.hf_model_loader import unwrap_hf_model

    _apply_state_dict(unwrap_hf_model(model), model_state, resolved_path, "model")
    _apply_state_dict(optimizer, optimizer_state, resolved_path, "optimizer")
    if scheduler is not None and scheduler_state is not None:
        _apply_state_dict(scheduler, scheduler_state, resolved_path, "scheduler")
    return CheckpointResumeState(
        next_epoch=epoch + 1,
        global_step=global_step,
        best_validation_loss=best_validation_loss,
        checkpoint_path=resolved_path,
    )


def _build_checkpoint_payload(
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    epoch: int,
    global_step: int,
    best_validation_loss: float | None,
) -> dict[str, object]:
    from serve.hf_model_loader import unwrap_hf_model

    payload = {
        "model_state_dict": unwrap_hf_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload


def _save_checkpoint_payload(torch_module: Any, checkpoint_path: Path, payload: object) -> None:
    try:
        torch_module.save(payload, str(checkpoint_path))
    except (OSError, RuntimeError) as error:
        raise CrucibleServeError(
            f"Failed to save checkpoint at {checkpoint_path}: {error}. "
            "Check available disk space and directory permissions."
        ) from error


def _read_checkpoint_payload(
    resolved_path: Path, torch_module: Any, device: Any
) -> Mapping[str, object]:
    if not resolved_path.exists():
        raise CrucibleServeError(
            f"Resume checkpoint not found at {resolved_path}. "
            "Provide a valid --resume-checkpoint-path value."
        )
    try:
        payload = torch_module.load(str(resolved_path), map_location=device, weights_only=True)
    except (OSError, RuntimeError) as error:
        raise CrucibleServeError(
            f"Failed to load checkpoint at {resolved_path}: {error}. "
            "Verify checkpoint file integrity and compatibility."
        ) from error
    if isinstance(payload, Mapping):
        return payload
    raise CrucibleServeError(
        f"Invalid checkpoint format at {resolved_path}: expected mapping payload."
    )


def _read_mapping(
    payload: Mapping[str, object], key: str, checkpoint_path: Path
) -> Mapping[str, object]:
    value = payload.get(key)
    if isinstance(value, Mapping):
        return value
    raise CrucibleServeError(
        f"Invalid checkpoint at {checkpoint_path}: missing mapping field '{key}'."
    )


def _read_epoch(payload: Mapping[str, object], checkpoint_path: Path) -> int:
    raw_epoch = payload.get("epoch")
    if isinstance(raw_epoch, int) and raw_epoch >= 1:
        return raw_epoch
    raise CrucibleServeError(
        f"Invalid checkpoint at {checkpoint_path}: field 'epoch' must be integer >= 1."
    )


def _read_global_step(payload: Mapping[str, object], checkpoint_path: Path) -> int:
    raw_global_step = payload.get("global_step")
    if isinstance(raw_global_step, int) and raw_global_step >= 0:
        return raw_global_step
    raise CrucibleServeError(
        f"Invalid checkpoint at {checkpoint_path}: field 'global_step' must be integer >= 0."
    )


def _read_optional_float(
    payload: Mapping[str, object],
    key: str,
    checkpoint_path: Path,
) -> float | None:
    raw_value = payload.get(key)
    if raw_value is None:
        return None
    if isinstance(raw_value, (float, int)):
        return float(raw_value)
    raise CrucibleServeError(
        f"Invalid checkpoint at {checkpoint_path}: field '{key}' must be numeric."
    )


def _read_optional_mapping(
    payload: Mapping[str, object],
    key: str,
    checkpoint_path: Path,
) -> Mapping[str, object] | None:
    raw_value = payload.get(key)
    if raw_value is None:
        return None
    if isinstance(raw_value, Mapping):
        return raw_value
    raise CrucibleServeError(
        f"Invalid checkpoint at {checkpoint_path}: field '{key}' must be a mapping."
    )


def _apply_state_dict(
    target: Any, state: Mapping[str, object], checkpoint_path: Path, target_name: str
) -> None:
    try:
        target.load_state_dict(state)
    except RuntimeError as error:
        raise CrucibleServeError(
            f"Failed to apply {target_name} state from {checkpoint_path}: {error}. "
            "Use a checkpoint created for the same model and optimizer setup."
        ) from error


def _list_epoch_checkpoints(checkpoint_dir: Path) -> list[Path]:
    epoch_files = []
    for candidate in checkpoint_dir.glob("epoch-*.pt"):
        epoch_index = _parse_epoch_index(candidate)
        if epoch_index is None:
            continue
        epoch_files.append((epoch_index, candidate))
    epoch_files.sort(key=lambda item: item[0])
    return [path for _, path in epoch_files]


def _parse_epoch_index(checkpoint_path: Path) -> int | None:
    file_name = checkpoint_path.stem
    if not file_name.startswith("epoch-"):
        return None
    raw_epoch = file_name.removeprefix("epoch-")
    if not raw_epoch.isdigit():
        return None
    return int(raw_epoch)


def _epoch_checkpoint_file_name(epoch: int) -> str:
    return f"epoch-{epoch:04d}.pt"
