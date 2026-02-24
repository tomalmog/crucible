"""Hyperparameter sweep orchestration.

This module runs multiple training trials with different hyperparameter
combinations and collects results to identify the best configuration.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.errors import ForgeSweepError
from core.sweep_types import SweepConfig, SweepResult, SweepTrialResult
from core.types import TrainingOptions, TrainingRunResult
from serve.sweep_analysis import find_best_trial, rank_trials
from serve.sweep_parameter_generator import generate_sweep_parameters
from store.dataset_sdk import ForgeClient


def run_sweep(
    client: ForgeClient,
    config: SweepConfig,
    random_seed: int,
) -> SweepResult:
    """Run a full hyperparameter sweep across parameter combinations.

    Args:
        client: SDK client for training execution.
        config: Sweep configuration with parameters and strategy.
        random_seed: Seed for reproducible random sampling.

    Returns:
        SweepResult with all trials and best configuration.

    Raises:
        ForgeSweepError: If no trials complete successfully.
    """
    param_combos = generate_sweep_parameters(config, random_seed)
    base_output = Path(config.base_output_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    trials: list[SweepTrialResult] = []
    for trial_id, params in enumerate(param_combos):
        trial_output = str(base_output / f"trial_{trial_id:04d}")
        result = _run_single_trial(
            client=client,
            config=config,
            trial_id=trial_id,
            params=params,
            trial_output=trial_output,
        )
        if result is not None:
            trials.append(result)
    return _build_sweep_result(trials, config.minimize)


def _run_single_trial(
    client: ForgeClient,
    config: SweepConfig,
    trial_id: int,
    params: dict[str, float],
    trial_output: str,
) -> SweepTrialResult | None:
    """Execute one training trial with overridden parameters.

    Args:
        client: SDK client for training execution.
        config: Sweep configuration.
        trial_id: Zero-based trial index.
        params: Parameter overrides for this trial.
        trial_output: Output directory for this trial.

    Returns:
        Trial result or None if trial failed.
    """
    try:
        options = _build_trial_options(config, params, trial_output)
        training_result = client.train(options)
        metric_value = _extract_metric(training_result, config.metric)
        return SweepTrialResult(
            trial_id=trial_id,
            parameters=params,
            metric_value=metric_value,
            model_path=training_result.model_path,
            history_path=training_result.history_path,
        )
    except Exception as exc:
        print(f"Trial {trial_id} failed: {exc}")
        return None


def _build_trial_options(
    config: SweepConfig,
    params: dict[str, float],
    trial_output: str,
) -> TrainingOptions:
    """Build TrainingOptions with sweep parameter overrides.

    Args:
        config: Sweep configuration for base options.
        params: Parameter name-to-value overrides.
        trial_output: Output directory for this trial.

    Returns:
        TrainingOptions with overridden hyperparameters.

    Raises:
        ForgeSweepError: If a parameter name is not a valid field.
    """
    base = TrainingOptions(
        dataset_name=config.dataset_name,
        output_dir=trial_output,
    )
    override_kwargs: dict[str, Any] = {}
    valid_fields = {f for f in base.__dataclass_fields__}
    for name, value in params.items():
        if name not in valid_fields:
            raise ForgeSweepError(
                f"Parameter '{name}' is not a valid TrainingOptions field."
            )
        field_type = _infer_field_type(name, base)
        override_kwargs[name] = field_type(value)
    return replace(base, **override_kwargs)


def _infer_field_type(name: str, base: TrainingOptions) -> type:
    """Infer the target type for a training option field.

    Args:
        name: Field name on TrainingOptions.
        base: Base TrainingOptions instance for type inspection.

    Returns:
        Python type to cast the parameter value to.
    """
    current_value = getattr(base, name)
    if isinstance(current_value, int):
        return int
    return float


def _extract_metric(
    result: TrainingRunResult,
    metric: str,
) -> float:
    """Extract a metric value from training result history.

    Args:
        result: Completed training run result.
        metric: Metric name to extract.

    Returns:
        Final metric value from training history.

    Raises:
        ForgeSweepError: If metric cannot be extracted.
    """
    history_path = Path(result.history_path)
    if not history_path.exists():
        raise ForgeSweepError(
            f"Training history not found at {history_path}."
        )
    try:
        history_data = json.loads(history_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ForgeSweepError(
            f"Failed to read training history: {exc}"
        ) from exc
    return _parse_metric_from_history(history_data, metric)


def _parse_metric_from_history(
    history_data: Any,
    metric: str,
) -> float:
    """Parse metric value from history JSON structure.

    Args:
        history_data: Parsed history JSON data.
        metric: Metric name to look up.

    Returns:
        Final epoch metric value.

    Raises:
        ForgeSweepError: If metric is not found in history.
    """
    epochs = history_data.get("epochs", [])
    if not epochs:
        raise ForgeSweepError(
            "Training history contains no epoch data."
        )
    last_epoch = epochs[-1]
    if metric not in last_epoch:
        available = ", ".join(sorted(last_epoch.keys()))
        raise ForgeSweepError(
            f"Metric '{metric}' not found in history. "
            f"Available: {available}."
        )
    value = last_epoch[metric]
    if not isinstance(value, (int, float)):
        raise ForgeSweepError(
            f"Metric '{metric}' value is not numeric: {value}."
        )
    return float(value)


def _build_sweep_result(
    trials: list[SweepTrialResult],
    minimize: bool,
) -> SweepResult:
    """Build final SweepResult from completed trials.

    Args:
        trials: List of completed trial results.
        minimize: Whether lower metric values are better.

    Returns:
        SweepResult with ranked trials and best configuration.

    Raises:
        ForgeSweepError: If no trials completed successfully.
    """
    if not trials:
        raise ForgeSweepError(
            "No sweep trials completed successfully. "
            "Check training configuration and dataset."
        )
    ranked = rank_trials(trials, minimize)
    best = find_best_trial(ranked)
    return SweepResult(
        trials=tuple(trials),
        best_trial_id=best.trial_id,
        best_parameters=best.parameters,
        best_metric_value=best.metric_value,
    )
