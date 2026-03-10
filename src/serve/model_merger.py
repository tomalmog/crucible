"""Model merging utilities.

This module supports merging multiple model weight files using
various strategies: SLERP, TIES, DARE, and weighted average.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from core.errors import CrucibleDependencyError

MergeMethod = Literal["slerp", "ties", "dare", "average"]

SUPPORTED_MERGE_METHODS: tuple[MergeMethod, ...] = ("slerp", "ties", "dare", "average")


@dataclass(frozen=True)
class MergeConfig:
    """Configuration for model merging.

    Attributes:
        model_paths: Paths to models to merge.
        method: Merge strategy.
        weights: Per-model weights (must sum to 1.0).
        output_path: Where to save merged model.
    """

    model_paths: tuple[str, ...]
    method: MergeMethod = "average"
    weights: tuple[float, ...] = ()
    output_path: str = "./merged_model.pt"


@dataclass(frozen=True)
class MergeResult:
    """Result of a model merge operation.

    Attributes:
        output_path: Path to merged model.
        method: Merge method used.
        num_models: Number of models merged.
        num_parameters: Total parameter count.
    """

    output_path: str
    method: str
    num_models: int
    num_parameters: int = 0


def merge_models(config: MergeConfig) -> MergeResult:
    """Merge multiple models using the specified strategy."""
    torch_module = _import_torch()
    state_dicts = []
    for path in config.model_paths:
        sd = torch_module.load(path, map_location="cpu", weights_only=True)
        state_dicts.append(sd)
    if not state_dicts:
        raise ValueError("No models provided for merging.")
    weights = config.weights or tuple(1.0 / len(state_dicts) for _ in state_dicts)
    if config.method == "average":
        merged = _weighted_average(state_dicts, weights)
    elif config.method == "slerp":
        merged = _slerp_merge(torch_module, state_dicts, weights)
    elif config.method == "ties":
        merged = _ties_merge(torch_module, state_dicts, weights)
    elif config.method == "dare":
        merged = _dare_merge(torch_module, state_dicts, weights)
    else:
        merged = _weighted_average(state_dicts, weights)
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(merged, output)
    num_params = sum(p.numel() for p in merged.values() if hasattr(p, "numel"))
    return MergeResult(
        output_path=str(output),
        method=config.method,
        num_models=len(state_dicts),
        num_parameters=num_params,
    )


def _weighted_average(
    state_dicts: list[dict[str, Any]],
    weights: tuple[float, ...],
) -> dict[str, Any]:
    """Merge via weighted average of parameters."""
    merged: dict[str, Any] = {}
    for key in state_dicts[0]:
        values = [sd[key].float() * w for sd, w in zip(state_dicts, weights)]
        merged[key] = sum(values[1:], values[0])
    return merged


def _slerp_merge(
    torch_module: Any,
    state_dicts: list[dict[str, Any]],
    weights: tuple[float, ...],
) -> dict[str, Any]:
    """Merge via spherical linear interpolation."""
    if len(state_dicts) == 2:
        t = weights[1] if len(weights) > 1 else 0.5
        merged: dict[str, Any] = {}
        for key in state_dicts[0]:
            v0 = state_dicts[0][key].float().flatten()
            v1 = state_dicts[1][key].float().flatten()
            norm0 = v0.norm()
            norm1 = v1.norm()
            if norm0 < 1e-8 or norm1 < 1e-8:
                merged[key] = ((1 - t) * state_dicts[0][key].float() + t * state_dicts[1][key].float())
                continue
            v0n = v0 / norm0
            v1n = v1 / norm1
            dot = torch_module.clamp(torch_module.dot(v0n, v1n), -1.0, 1.0)
            omega = torch_module.acos(dot)
            if omega.abs() < 1e-8:
                merged[key] = ((1 - t) * state_dicts[0][key].float() + t * state_dicts[1][key].float())
            else:
                sin_omega = torch_module.sin(omega)
                s0 = torch_module.sin((1 - t) * omega) / sin_omega
                s1 = torch_module.sin(t * omega) / sin_omega
                result = s0 * v0 + s1 * v1
                merged[key] = result.view(state_dicts[0][key].shape)
        return merged
    return _weighted_average(state_dicts, weights)


def _ties_merge(
    torch_module: Any,
    state_dicts: list[dict[str, Any]],
    weights: tuple[float, ...],
) -> dict[str, Any]:
    """Merge via TIES (TrIm, Elect Sign, and merge)."""
    base = state_dicts[0]
    merged: dict[str, Any] = {}
    for key in base:
        deltas = [(sd[key].float() - base[key].float()) * w for sd, w in zip(state_dicts[1:], weights[1:])]
        if not deltas:
            merged[key] = base[key].float()
            continue
        stacked = torch_module.stack(deltas)
        signs = torch_module.sign(stacked.sum(dim=0))
        mask = (torch_module.sign(stacked) == signs.unsqueeze(0))
        masked = stacked * mask.float()
        avg_delta = masked.sum(dim=0) / mask.float().sum(dim=0).clamp(min=1)
        merged[key] = base[key].float() + avg_delta
    return merged


def _dare_merge(
    torch_module: Any,
    state_dicts: list[dict[str, Any]],
    weights: tuple[float, ...],
    drop_rate: float = 0.1,
) -> dict[str, Any]:
    """Merge via DARE (Drop And REscale)."""
    base = state_dicts[0]
    merged: dict[str, Any] = {}
    for key in base:
        deltas = [(sd[key].float() - base[key].float()) for sd in state_dicts[1:]]
        if not deltas:
            merged[key] = base[key].float()
            continue
        rescaled = []
        for delta, w in zip(deltas, weights[1:]):
            mask = (torch_module.rand_like(delta.float()) > drop_rate).float()
            rescaled.append(delta * mask * w / (1.0 - drop_rate))
        merged[key] = base[key].float() + sum(rescaled[1:], rescaled[0])
    return merged


def _import_torch() -> Any:
    """Import torch dependency."""
    try:
        import torch
        return torch
    except ImportError as error:
        raise CrucibleDependencyError(
            "Model merging requires torch. Install torch to run crucible merge."
        ) from error
