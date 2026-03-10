"""Optimizer and scheduler builders for training runs.

This module maps typed training options to torch optimizer and scheduler
instances so training behavior can be configured through CLI and YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.errors import CrucibleServeError
from core.types import TrainingOptions


@dataclass(frozen=True)
class TrainingOptimization:
    """Runtime optimization objects used by the training loop."""

    optimizer: Any
    scheduler: Any | None


def build_training_optimization(
    torch_module: Any,
    model: Any,
    options: TrainingOptions,
) -> TrainingOptimization:
    """Build optimizer and optional scheduler from training options."""
    optimizer = _build_optimizer(torch_module, model, options)
    scheduler = _build_scheduler(torch_module, optimizer, options)
    return TrainingOptimization(optimizer=optimizer, scheduler=scheduler)


def _build_optimizer(torch_module: Any, model: Any, options: TrainingOptions) -> Any:
    """Build optimizer instance from configured optimizer type."""
    parameters = model.parameters()
    if options.optimizer_type == "adam":
        return torch_module.optim.Adam(
            parameters,
            lr=options.learning_rate,
            weight_decay=options.weight_decay,
        )
    if options.optimizer_type == "adamw":
        return torch_module.optim.AdamW(
            parameters,
            lr=options.learning_rate,
            weight_decay=options.weight_decay,
        )
    if options.optimizer_type == "sgd":
        return torch_module.optim.SGD(
            parameters,
            lr=options.learning_rate,
            momentum=options.sgd_momentum,
            weight_decay=options.weight_decay,
        )
    raise CrucibleServeError(
        f"Unsupported optimizer_type {options.optimizer_type!r}. "
        "Use adam, adamw, or sgd."
    )


def _build_scheduler(torch_module: Any, optimizer: Any, options: TrainingOptions) -> Any | None:
    """Build optional scheduler instance from configured scheduler type."""
    if options.scheduler_type == "none":
        return None
    if options.scheduler_type == "step":
        return torch_module.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=options.scheduler_step_size,
            gamma=options.scheduler_gamma,
        )
    if options.scheduler_type == "cosine":
        t_max_epochs = options.scheduler_t_max_epochs or options.epochs
        return torch_module.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max_epochs,
            eta_min=options.scheduler_eta_min,
        )
    raise CrucibleServeError(
        f"Unsupported scheduler_type {options.scheduler_type!r}. "
        "Use none, step, or cosine."
    )
