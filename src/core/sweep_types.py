"""Typed models for hyperparameter sweep configuration and results.

This module defines immutable data models used by sweep runners,
parameter generators, and CLI commands to keep interfaces explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SweepStrategy = Literal["grid", "random"]


@dataclass(frozen=True)
class SweepParameter:
    """One hyperparameter to sweep.

    Attributes:
        name: Field name in TrainingOptions (e.g., "learning_rate").
        values: Explicit values for grid search.
        min_value: Lower bound for random search.
        max_value: Upper bound for random search.
        log_scale: Sample in log space for random search.
    """

    name: str
    values: tuple[float, ...] = ()
    min_value: float = 0.0
    max_value: float = 1.0
    log_scale: bool = False


@dataclass(frozen=True)
class SweepConfig:
    """Hyperparameter sweep configuration.

    Attributes:
        dataset_name: Name of the dataset to train on.
        output_dir: Output directory for training artifacts.
        base_output_dir: Parent directory for all trial outputs.
        parameters: Tuple of hyperparameters to sweep.
        strategy: Search strategy (grid or random).
        max_trials: Maximum number of trials for random search.
        metric: Metric name to optimize.
        minimize: True if lower metric values are better.
    """

    dataset_name: str
    output_dir: str
    base_output_dir: str
    parameters: tuple[SweepParameter, ...] = ()
    strategy: SweepStrategy = "grid"
    max_trials: int = 10
    metric: str = "validation_loss"
    minimize: bool = True


@dataclass(frozen=True)
class SweepTrialResult:
    """Result of one sweep trial.

    Attributes:
        trial_id: Zero-based trial index.
        parameters: Mapping of parameter names to values used.
        metric_value: Observed metric value for this trial.
        model_path: Path to saved model weights.
        history_path: Path to training history JSON.
    """

    trial_id: int
    parameters: dict[str, float]
    metric_value: float
    model_path: str
    history_path: str


@dataclass(frozen=True)
class SweepResult:
    """Full sweep result across all trials.

    Attributes:
        trials: All completed trial results.
        best_trial_id: Trial ID with the best metric.
        best_parameters: Parameters from the best trial.
        best_metric_value: Best observed metric value.
    """

    trials: tuple[SweepTrialResult, ...]
    best_trial_id: int
    best_parameters: dict[str, float]
    best_metric_value: float
