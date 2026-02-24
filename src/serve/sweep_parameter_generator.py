"""Hyperparameter sweep parameter generation.

This module generates parameter combinations for grid and random
sweep strategies, used by the sweep runner to orchestrate trials.
"""

from __future__ import annotations

import itertools
import math
import random as random_module
from typing import Sequence

from core.errors import ForgeSweepError
from core.sweep_types import SweepConfig, SweepParameter


def generate_sweep_parameters(
    config: SweepConfig,
    random_seed: int,
) -> list[dict[str, float]]:
    """Dispatch parameter generation based on sweep strategy.

    Args:
        config: Sweep configuration with strategy and parameters.
        random_seed: Seed for reproducible random sampling.

    Returns:
        List of parameter dictionaries for each trial.

    Raises:
        ForgeSweepError: If strategy is unsupported or parameters invalid.
    """
    _validate_parameters(config.parameters)
    if config.strategy == "grid":
        return generate_grid_parameters(config.parameters)
    if config.strategy == "random":
        return generate_random_parameters(
            config.parameters,
            config.max_trials,
            random_seed,
        )
    raise ForgeSweepError(
        f"Unsupported sweep strategy '{config.strategy}'. Use 'grid' or 'random'."
    )


def generate_grid_parameters(
    parameters: tuple[SweepParameter, ...],
) -> list[dict[str, float]]:
    """Generate cartesian product of all parameter values.

    Args:
        parameters: Parameters with explicit value lists.

    Returns:
        List of parameter dictionaries, one per grid point.

    Raises:
        ForgeSweepError: If any parameter has no values for grid search.
    """
    for param in parameters:
        if not param.values:
            raise ForgeSweepError(
                f"Grid search requires explicit 'values' for parameter "
                f"'{param.name}'. Provide at least one value."
            )
    names = [p.name for p in parameters]
    value_lists = [p.values for p in parameters]
    combinations = list(itertools.product(*value_lists))
    return [dict(zip(names, combo)) for combo in combinations]


def generate_random_parameters(
    parameters: tuple[SweepParameter, ...],
    max_trials: int,
    random_seed: int,
) -> list[dict[str, float]]:
    """Generate random parameter samples within defined bounds.

    Args:
        parameters: Parameters with min/max bounds.
        max_trials: Number of random trials to generate.
        random_seed: Seed for reproducibility.

    Returns:
        List of parameter dictionaries, one per trial.

    Raises:
        ForgeSweepError: If min_value >= max_value for any parameter.
    """
    _validate_random_bounds(parameters)
    rng = random_module.Random(random_seed)
    results: list[dict[str, float]] = []
    for _ in range(max_trials):
        combo: dict[str, float] = {}
        for param in parameters:
            combo[param.name] = _sample_value(rng, param)
        results.append(combo)
    return results


def _sample_value(
    rng: random_module.Random,
    param: SweepParameter,
) -> float:
    """Sample a single parameter value using uniform or log-uniform.

    Args:
        rng: Random number generator instance.
        param: Parameter specification with bounds and scale.

    Returns:
        Sampled float value.
    """
    if param.values:
        return rng.choice(param.values)
    if param.log_scale:
        return _sample_log_uniform(rng, param.min_value, param.max_value)
    return rng.uniform(param.min_value, param.max_value)


def _sample_log_uniform(
    rng: random_module.Random,
    min_value: float,
    max_value: float,
) -> float:
    """Sample from log-uniform distribution between bounds.

    Args:
        rng: Random number generator instance.
        min_value: Lower bound (must be positive).
        max_value: Upper bound (must be positive).

    Returns:
        Log-uniformly sampled value.

    Raises:
        ForgeSweepError: If bounds are not positive for log scale.
    """
    if min_value <= 0 or max_value <= 0:
        raise ForgeSweepError(
            "Log-scale sampling requires positive min_value and max_value."
        )
    log_min = math.log(min_value)
    log_max = math.log(max_value)
    return math.exp(rng.uniform(log_min, log_max))


def _validate_parameters(
    parameters: tuple[SweepParameter, ...],
) -> None:
    """Validate that at least one parameter is provided.

    Args:
        parameters: Sweep parameters to validate.

    Raises:
        ForgeSweepError: If parameters tuple is empty.
    """
    if not parameters:
        raise ForgeSweepError(
            "Sweep requires at least one parameter. Add parameters to config."
        )


def _validate_random_bounds(
    parameters: Sequence[SweepParameter],
) -> None:
    """Validate random search bounds for each parameter.

    Args:
        parameters: Parameters to validate.

    Raises:
        ForgeSweepError: If min_value >= max_value for non-values params.
    """
    for param in parameters:
        if param.values:
            continue
        if param.min_value >= param.max_value:
            raise ForgeSweepError(
                f"Parameter '{param.name}' has min_value >= max_value "
                f"({param.min_value} >= {param.max_value}). "
                f"Set min_value < max_value for random search."
            )
