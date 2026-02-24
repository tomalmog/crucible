"""Unit tests for hyperparameter sweep result analysis."""

from __future__ import annotations

from core.sweep_types import SweepResult, SweepTrialResult
from serve.sweep_analysis import (
    format_sweep_report,
    rank_trials,
    top_k_trials,
)


def _build_trials() -> list[SweepTrialResult]:
    """Create sample trial results for testing."""
    return [
        SweepTrialResult(
            trial_id=0,
            parameters={"learning_rate": 0.01},
            metric_value=0.5,
            model_path="/tmp/trial_0/model.pt",
            history_path="/tmp/trial_0/history.json",
        ),
        SweepTrialResult(
            trial_id=1,
            parameters={"learning_rate": 0.001},
            metric_value=0.3,
            model_path="/tmp/trial_1/model.pt",
            history_path="/tmp/trial_1/history.json",
        ),
        SweepTrialResult(
            trial_id=2,
            parameters={"learning_rate": 0.1},
            metric_value=0.8,
            model_path="/tmp/trial_2/model.pt",
            history_path="/tmp/trial_2/history.json",
        ),
    ]


def _build_sweep_result() -> SweepResult:
    """Create a sample SweepResult for testing."""
    trials = _build_trials()
    return SweepResult(
        trials=tuple(trials),
        best_trial_id=1,
        best_parameters={"learning_rate": 0.001},
        best_metric_value=0.3,
    )


def test_rank_trials_minimize() -> None:
    """Ranking with minimize=True should sort ascending."""
    trials = _build_trials()
    ranked = rank_trials(trials, minimize=True)

    assert ranked[0].trial_id == 1
    assert ranked[1].trial_id == 0
    assert ranked[2].trial_id == 2


def test_rank_trials_maximize() -> None:
    """Ranking with minimize=False should sort descending."""
    trials = _build_trials()
    ranked = rank_trials(trials, minimize=False)

    assert ranked[0].trial_id == 2
    assert ranked[1].trial_id == 0
    assert ranked[2].trial_id == 1


def test_rank_trials_single() -> None:
    """Ranking a single trial should return it unchanged."""
    trials = [_build_trials()[0]]
    ranked = rank_trials(trials, minimize=True)

    assert len(ranked) == 1
    assert ranked[0].trial_id == 0


def test_top_k_trials_returns_k() -> None:
    """top_k should return the requested number of trials."""
    result = _build_sweep_result()
    top = top_k_trials(result, k=2)

    assert len(top) == 2


def test_top_k_trials_best_first() -> None:
    """top_k should return the best trial first."""
    result = _build_sweep_result()
    top = top_k_trials(result, k=1)

    assert len(top) == 1
    assert top[0].trial_id == 1


def test_top_k_trials_k_exceeds_count() -> None:
    """top_k with k > total trials should return all trials."""
    result = _build_sweep_result()
    top = top_k_trials(result, k=100)

    assert len(top) == 3


def test_format_sweep_report_contains_header() -> None:
    """Report should include a header line."""
    result = _build_sweep_result()
    report = format_sweep_report(result)

    assert "Hyperparameter Sweep Report" in report


def test_format_sweep_report_contains_best_info() -> None:
    """Report should include best trial information."""
    result = _build_sweep_result()
    report = format_sweep_report(result)

    assert "Best trial" in report
    assert "#1" in report
    assert "0.300000" in report


def test_format_sweep_report_contains_trial_count() -> None:
    """Report should show total trial count."""
    result = _build_sweep_result()
    report = format_sweep_report(result)

    assert "Total trials: 3" in report


def test_format_sweep_report_contains_parameters() -> None:
    """Report should list best parameters."""
    result = _build_sweep_result()
    report = format_sweep_report(result)

    assert "learning_rate" in report


def test_format_sweep_report_lists_all_trials() -> None:
    """Report should list all trial metric values."""
    result = _build_sweep_result()
    report = format_sweep_report(result)

    assert "0.500000" in report
    assert "0.300000" in report
    assert "0.800000" in report


def test_rank_trials_preserves_data() -> None:
    """Ranking should not modify trial data."""
    trials = _build_trials()
    ranked = rank_trials(trials, minimize=True)

    best = ranked[0]
    assert best.parameters == {"learning_rate": 0.001}
    assert best.model_path == "/tmp/trial_1/model.pt"
    assert best.history_path == "/tmp/trial_1/history.json"
