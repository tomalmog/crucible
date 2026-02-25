"""End-to-end integration tests for model merging.

Tests train real tiny models with PyTorch and merge them using
real merge_models() — no mocked torch dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cli.main import main
from core.types import DataRecord, RecordMetadata, TrainingOptions
from serve.model_merger import MergeConfig, MergeResult, merge_models
from serve.training_runner import run_training

# ── Shared tiny training config ──────────────────────────────────────
_TINY = dict(
    epochs=1,
    batch_size=2,
    hidden_dim=32,
    num_layers=1,
    attention_heads=2,
    max_token_length=64,
    learning_rate=0.001,
    validation_split=0.2,
)


def _records() -> list[DataRecord]:
    return [
        DataRecord(
            record_id=f"train-{i}",
            text="hello world test sentence " * 10,
            metadata=RecordMetadata("test", "en", 0.9, 100.0),
        )
        for i in range(5)
    ]


def _train_model(tmp_path: Path, name: str) -> str:
    """Train a tiny model and return its model.pt path."""
    options = TrainingOptions(
        dataset_name="test",
        output_dir=str(tmp_path / name),
        **_TINY,
    )
    result = run_training(_records(), options, 42, tmp_path, "v1")
    return result.model_path


def test_config_creation() -> None:
    """MergeConfig should store method, model_paths, and defaults."""
    for method in ("slerp", "ties", "dare", "average"):
        cfg = MergeConfig(
            model_paths=("a.pt", "b.pt"),
            method=method,  # type: ignore[arg-type]
        )
        assert cfg.method == method
        assert cfg.model_paths == ("a.pt", "b.pt")
        assert cfg.output_path == "./merged_model.pt"
        assert cfg.weights == ()


def test_weighted_config() -> None:
    """MergeConfig should preserve explicit weights."""
    cfg = MergeConfig(
        model_paths=("a.pt", "b.pt"),
        method="average",
        weights=(0.7, 0.3),
    )
    assert cfg.weights == (0.7, 0.3)
    assert cfg.model_paths == ("a.pt", "b.pt")


def test_merge_average_real(tmp_path: Path) -> None:
    """Train 2 models, merge with 'average', verify output is loadable."""
    path_a = _train_model(tmp_path, "model_a")
    path_b = _train_model(tmp_path, "model_b")
    output = str(tmp_path / "merged_avg.pt")

    config = MergeConfig(
        model_paths=(path_a, path_b),
        method="average",
        output_path=output,
    )
    result = merge_models(config)

    assert isinstance(result, MergeResult)
    assert result.method == "average"
    assert result.num_models == 2
    assert Path(result.output_path).exists()
    merged_state = torch.load(output, map_location="cpu", weights_only=True)
    original_state = torch.load(path_a, map_location="cpu", weights_only=True)
    assert set(merged_state.keys()) == set(original_state.keys())


def test_merge_slerp_real(tmp_path: Path) -> None:
    """Train 2 models, merge with 'slerp', verify output is loadable."""
    path_a = _train_model(tmp_path, "model_a")
    path_b = _train_model(tmp_path, "model_b")
    output = str(tmp_path / "merged_slerp.pt")

    config = MergeConfig(
        model_paths=(path_a, path_b),
        method="slerp",
        output_path=output,
    )
    result = merge_models(config)

    assert isinstance(result, MergeResult)
    assert result.method == "slerp"
    assert result.num_models == 2
    assert Path(result.output_path).exists()
    merged_state = torch.load(output, map_location="cpu", weights_only=True)
    original_state = torch.load(path_a, map_location="cpu", weights_only=True)
    assert set(merged_state.keys()) == set(original_state.keys())


def test_cli_merge_real(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI merge with real trained models should exit 0."""
    path_a = _train_model(tmp_path, "model_a")
    path_b = _train_model(tmp_path, "model_b")
    output = str(tmp_path / "cli_merged.pt")

    exit_code = main([
        "--data-root", str(tmp_path),
        "merge",
        "--models", path_a, path_b,
        "--method", "average",
        "--output", output,
    ])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "method=average" in captured
    assert "num_models=2" in captured
    assert Path(output).exists()
