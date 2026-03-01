"""Hyperparameter sweep orchestration.

This module runs multiple training trials with different hyperparameter
combinations and collects results to identify the best configuration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from core.errors import ForgeSweepError
from core.sweep_types import SweepConfig, SweepResult, SweepTrialResult
from core.training_methods import TRAINING_METHOD_DISPATCH, TRAINING_METHOD_LABELS, dispatch_training
from core.training_types import TrainingRunResult
from serve.sweep_analysis import find_best_trial, rank_trials
from serve.sweep_parameter_generator import generate_sweep_parameters
from store.dataset_sdk import ForgeClient


def _log(message: str) -> None:
    """Print a progress message and flush immediately for piped output."""
    print(message, flush=True)


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
    total_trials = len(param_combos)
    method_label = TRAINING_METHOD_LABELS.get(config.training_method, config.training_method)
    base_output = Path(config.base_output_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    _log(f"Sweep: {total_trials} trials | method={method_label} | strategy={config.strategy} | metric={config.metric}")
    trials: list[SweepTrialResult] = []
    errors: list[str] = []
    for trial_id, params in enumerate(param_combos):
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        _log(f"Trial {trial_id + 1}/{total_trials} | {params_str} | starting...")
        trial_output = str(base_output / f"trial_{trial_id:04d}")
        result, error = _run_single_trial(
            client=client,
            config=config,
            trial_id=trial_id,
            params=params,
            trial_output=trial_output,
        )
        if result is not None:
            trials.append(result)
            _log(f"Trial {trial_id + 1}/{total_trials} | {params_str} | {config.metric}={result.metric_value:.6f}")
        else:
            errors.append(error or "unknown error")
            _log(f"Trial {trial_id + 1}/{total_trials} | {params_str} | FAILED: {error}")
    _log(f"Sweep complete: {len(trials)}/{total_trials} trials succeeded")
    return _build_sweep_result(trials, config.minimize, errors)


def _run_single_trial(
    client: ForgeClient,
    config: SweepConfig,
    trial_id: int,
    params: dict[str, float],
    trial_output: str,
) -> tuple[SweepTrialResult | None, str | None]:
    """Execute one training trial with overridden parameters.

    Args:
        client: SDK client for training execution.
        config: Sweep configuration.
        trial_id: Zero-based trial index.
        params: Parameter overrides for this trial.
        trial_output: Output directory for this trial.

    Returns:
        Tuple of (trial result, error message). One will be None.
    """
    try:
        kwargs = _build_trial_kwargs(config, params, trial_output)
        training_result = dispatch_training(
            client, config.training_method, kwargs,
        )
        metric_value = _extract_metric(training_result, config.metric)
        return SweepTrialResult(
            trial_id=trial_id,
            parameters=params,
            metric_value=metric_value,
            model_path=training_result.model_path,
            history_path=training_result.history_path,
        ), None
    except Exception as exc:
        return None, str(exc)


def _build_trial_kwargs(
    config: SweepConfig,
    params: dict[str, float],
    trial_output: str,
) -> dict[str, Any]:
    """Build keyword arguments for the training options dataclass.

    Merges base config, method-specific fixed args, and swept parameters.

    Args:
        config: Sweep configuration.
        params: Swept parameter overrides for this trial.
        trial_output: Output directory for this trial.

    Returns:
        Keyword arguments dict for the target options class.
    """
    method = config.training_method
    if method not in TRAINING_METHOD_DISPATCH:
        raise ForgeSweepError(
            f"Unknown training method '{method}'."
        )
    _, options_class = TRAINING_METHOD_DISPATCH[method]
    kwargs: dict[str, Any] = {
        "dataset_name": config.dataset_name,
        "output_dir": trial_output,
    }
    # Apply fixed method-specific args (e.g. sft_data_path, base_model)
    for key, value in config.method_args:
        kwargs[key] = value
    # Apply swept parameters with type casting
    valid_fields = set(options_class.__dataclass_fields__)
    for name, value in params.items():
        if name not in valid_fields:
            raise ForgeSweepError(
                f"Parameter '{name}' is not a valid field for "
                f"{options_class.__name__}."
            )
        kwargs[name] = _cast_field_value(name, value, options_class)
    return kwargs


def _cast_field_value(
    name: str,
    value: float,
    options_class: type,
) -> int | float:
    """Cast a swept parameter value to the correct type for the options class.

    Args:
        name: Field name.
        value: Float value from sweep parameters.
        options_class: Target dataclass type.

    Returns:
        Value cast to int or float as appropriate.
    """
    field_info = options_class.__dataclass_fields__.get(name)
    if field_info is not None and field_info.type in ("int", int):
        return int(value)
    return value


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
    errors: list[str] | None = None,
) -> SweepResult:
    """Build final SweepResult from completed trials.

    Args:
        trials: List of completed trial results.
        minimize: Whether lower metric values are better.
        errors: Error messages from failed trials.

    Returns:
        SweepResult with ranked trials and best configuration.

    Raises:
        ForgeSweepError: If no trials completed successfully.
    """
    if not trials:
        msg = "No sweep trials completed successfully."
        if errors:
            first_error = errors[0]
            msg += f" First trial error: {first_error}"
        raise ForgeSweepError(msg)
    ranked = rank_trials(trials, minimize)
    best = find_best_trial(ranked)
    return SweepResult(
        trials=tuple(trials),
        best_trial_id=best.trial_id,
        best_parameters=best.parameters,
        best_metric_value=best.metric_value,
    )
