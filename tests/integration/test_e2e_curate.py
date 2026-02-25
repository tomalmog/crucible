"""End-to-end integration tests for the dataset curation workflow.

Tests cover the quality scoring engine, distribution analysis,
and CLI entry points for the curate command.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cli.main import main
from serve.dataset_curator import compute_distributions, score_examples


def test_score_good_content() -> None:
    """A reasonably long document should receive a perfect score with no issues."""
    results = score_examples([
        {"id": "r1", "text": "A reasonably long document with enough content to pass all quality checks easily."},
    ])

    assert len(results) == 1
    assert results[0].score == 1.0
    assert len(results[0].issues) == 0


def test_score_short() -> None:
    """A very short text should be flagged as too_short with a reduced score."""
    results = score_examples([{"id": "r1", "text": "hi"}])

    assert len(results) == 1
    assert results[0].score < 1.0
    assert "too_short" in results[0].issues


def test_score_empty() -> None:
    """An empty text should receive a zero score and be flagged as empty_content."""
    results = score_examples([{"id": "r1", "text": ""}])

    assert len(results) == 1
    assert results[0].score == 0.0
    assert "empty_content" in results[0].issues


def test_score_repetitive() -> None:
    """Highly repetitive text should be flagged as highly_repetitive."""
    results = score_examples([{"id": "r1", "text": "word " * 100}])

    assert len(results) == 1
    assert "highly_repetitive" in results[0].issues


def test_distributions() -> None:
    """Distribution stats should reflect correct totals and token length bounds."""
    records = [
        {"id": "r1", "text": "short text"},
        {"id": "r2", "text": " ".join(["medium"] * 50)},
        {"id": "r3", "text": " ".join(["longer"] * 150)},
    ]
    dist = compute_distributions(records)

    assert dist.total_records == 3
    assert dist.min_token_length == 2
    assert dist.max_token_length == 150
    assert dist.avg_token_length == pytest.approx((2 + 50 + 150) / 3, rel=0.01)


def test_cli_score(
    ingested_dataset: tuple,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI curate score should exit 0 when run against an ingested dataset."""
    client, dataset_name, _ = ingested_dataset
    data_root = str(client._config.data_root)

    exit_code = main(["--data-root", data_root, "curate", "score", "--dataset", dataset_name])
    captured = capsys.readouterr().out

    assert exit_code == 0


def test_cli_stats(
    ingested_dataset: tuple,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI curate stats should exit 0 and print total_records."""
    client, dataset_name, _ = ingested_dataset
    data_root = str(client._config.data_root)

    exit_code = main(["--data-root", data_root, "curate", "stats", "--dataset", dataset_name])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "total_records" in captured
