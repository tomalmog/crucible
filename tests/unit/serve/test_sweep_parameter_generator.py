"""Unit tests for hyperparameter sweep parameter generation."""

from __future__ import annotations

import math

import pytest

from core.errors import CrucibleSweepError
from core.sweep_types import SweepConfig, SweepParameter
from serve.sweep_parameter_generator import (
    generate_grid_parameters,
    generate_random_parameters,
    generate_sweep_parameters,
)


def test_grid_generates_cartesian_product() -> None:
    """Grid search should produce all combinations of parameter values."""
    params = (
        SweepParameter(name="learning_rate", values=(0.01, 0.001)),
        SweepParameter(name="dropout", values=(0.1, 0.2, 0.3)),
    )
    result = generate_grid_parameters(params)

    assert len(result) == 6
    assert {"learning_rate": 0.01, "dropout": 0.1} in result
    assert {"learning_rate": 0.01, "dropout": 0.2} in result
    assert {"learning_rate": 0.01, "dropout": 0.3} in result
    assert {"learning_rate": 0.001, "dropout": 0.1} in result
    assert {"learning_rate": 0.001, "dropout": 0.2} in result
    assert {"learning_rate": 0.001, "dropout": 0.3} in result


def test_grid_single_parameter() -> None:
    """Grid with one parameter should produce one dict per value."""
    params = (SweepParameter(name="epochs", values=(5.0, 10.0, 15.0)),)
    result = generate_grid_parameters(params)

    assert len(result) == 3
    assert result[0] == {"epochs": 5.0}
    assert result[1] == {"epochs": 10.0}
    assert result[2] == {"epochs": 15.0}


def test_grid_raises_on_empty_values() -> None:
    """Grid search should fail if any parameter has no values."""
    params = (SweepParameter(name="learning_rate"),)
    with pytest.raises(CrucibleSweepError, match="explicit 'values'"):
        generate_grid_parameters(params)


def test_random_generates_correct_count() -> None:
    """Random search should generate exactly max_trials combinations."""
    params = (
        SweepParameter(
            name="learning_rate", min_value=0.0001, max_value=0.1,
        ),
    )
    result = generate_random_parameters(params, max_trials=5, random_seed=42)

    assert len(result) == 5


def test_random_values_within_bounds() -> None:
    """Random samples should fall within specified bounds."""
    params = (
        SweepParameter(
            name="learning_rate", min_value=0.001, max_value=0.1,
        ),
        SweepParameter(
            name="dropout", min_value=0.0, max_value=0.5,
        ),
    )
    result = generate_random_parameters(params, max_trials=100, random_seed=7)

    for combo in result:
        assert 0.001 <= combo["learning_rate"] <= 0.1
        assert 0.0 <= combo["dropout"] <= 0.5


def test_random_log_scale_within_bounds() -> None:
    """Log-scale samples should fall within specified bounds."""
    params = (
        SweepParameter(
            name="learning_rate",
            min_value=0.0001,
            max_value=0.1,
            log_scale=True,
        ),
    )
    result = generate_random_parameters(params, max_trials=100, random_seed=42)

    for combo in result:
        assert 0.0001 <= combo["learning_rate"] <= 0.1


def test_random_log_scale_distribution() -> None:
    """Log-scale should sample more densely near lower bound."""
    params = (
        SweepParameter(
            name="learning_rate",
            min_value=0.0001,
            max_value=1.0,
            log_scale=True,
        ),
    )
    result = generate_random_parameters(params, max_trials=1000, random_seed=42)
    values = [combo["learning_rate"] for combo in result]
    below_median = sum(1 for v in values if v < 0.01)

    assert below_median > len(values) * 0.3


def test_random_reproducible_with_same_seed() -> None:
    """Same random seed should produce identical results."""
    params = (
        SweepParameter(
            name="learning_rate", min_value=0.001, max_value=0.1,
        ),
    )
    result_a = generate_random_parameters(params, max_trials=5, random_seed=42)
    result_b = generate_random_parameters(params, max_trials=5, random_seed=42)

    assert result_a == result_b


def test_random_raises_on_invalid_bounds() -> None:
    """Random search should fail when min_value >= max_value."""
    params = (
        SweepParameter(
            name="learning_rate", min_value=0.1, max_value=0.1,
        ),
    )
    with pytest.raises(CrucibleSweepError, match="min_value >= max_value"):
        generate_random_parameters(params, max_trials=5, random_seed=42)


def test_log_scale_raises_on_non_positive_bounds() -> None:
    """Log-scale should fail with non-positive bounds."""
    params = (
        SweepParameter(
            name="learning_rate",
            min_value=-0.1,
            max_value=0.1,
            log_scale=True,
        ),
    )
    with pytest.raises(CrucibleSweepError, match="positive"):
        generate_random_parameters(params, max_trials=1, random_seed=42)


def test_generate_sweep_parameters_dispatches_grid() -> None:
    """Dispatch should call grid generation for grid strategy."""
    config = SweepConfig(
        dataset_name="demo",
        output_dir="/tmp/out",
        base_output_dir="/tmp/out",
        parameters=(
            SweepParameter(name="learning_rate", values=(0.01, 0.001)),
        ),
        strategy="grid",
    )
    result = generate_sweep_parameters(config, random_seed=42)

    assert len(result) == 2


def test_generate_sweep_parameters_dispatches_random() -> None:
    """Dispatch should call random generation for random strategy."""
    config = SweepConfig(
        dataset_name="demo",
        output_dir="/tmp/out",
        base_output_dir="/tmp/out",
        parameters=(
            SweepParameter(
                name="learning_rate", min_value=0.001, max_value=0.1,
            ),
        ),
        strategy="random",
        max_trials=3,
    )
    result = generate_sweep_parameters(config, random_seed=42)

    assert len(result) == 3


def test_generate_sweep_parameters_raises_on_empty() -> None:
    """Dispatch should fail when no parameters are provided."""
    config = SweepConfig(
        dataset_name="demo",
        output_dir="/tmp/out",
        base_output_dir="/tmp/out",
        parameters=(),
        strategy="grid",
    )
    with pytest.raises(CrucibleSweepError, match="at least one parameter"):
        generate_sweep_parameters(config, random_seed=42)


def test_random_with_explicit_values_choices() -> None:
    """Random search with explicit values should choose from them."""
    params = (
        SweepParameter(
            name="learning_rate", values=(0.01, 0.001, 0.0001),
        ),
    )
    result = generate_random_parameters(params, max_trials=10, random_seed=42)

    for combo in result:
        assert combo["learning_rate"] in (0.01, 0.001, 0.0001)
