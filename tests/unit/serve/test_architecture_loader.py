"""Unit tests for architecture loader behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import serve.architecture_loader as architecture_loader
from core.errors import CrucibleServeError
from core.types import TrainingOptions
from serve.architecture_loader import load_training_model


class _FakeTorch:
    class nn:
        class Module:
            pass


def test_load_training_model_raises_for_missing_architecture_file(tmp_path: Path) -> None:
    """Missing architecture file should raise serve error."""
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        architecture_path=str(tmp_path / "missing.py"),
    )

    with pytest.raises(CrucibleServeError):
        load_training_model(_FakeTorch(), options, vocab_size=10)

    assert True


def test_load_training_model_reads_json_default_architecture(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """JSON default architecture config should load model instance."""
    captured: dict[str, object] = {}

    def _fake_build_default_model(torch_module, vocab_size, options):
        captured["vocab_size"] = vocab_size
        captured["hidden_dim"] = options.hidden_dim
        captured["num_layers"] = options.num_layers
        captured["attention_heads"] = options.attention_heads
        captured["mlp_hidden_dim"] = options.mlp_hidden_dim
        captured["mlp_layers"] = options.mlp_layers
        captured["dropout"] = options.dropout
        captured["position_embedding_type"] = options.position_embedding_type
        return _FakeTorch.nn.Module()

    monkeypatch.setattr(architecture_loader, "build_default_model", _fake_build_default_model)
    architecture_path = tmp_path / "architecture.json"
    architecture_path.write_text(
        json.dumps(
            {
                "architecture": "default",
                "hidden_dim": 8,
                "num_layers": 1,
                "attention_heads": 2,
                "mlp_hidden_dim": 64,
                "mlp_layers": 3,
                "dropout": 0.2,
                "position_embedding_type": "sinusoidal",
            }
        ),
        encoding="utf-8",
    )
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        architecture_path=str(architecture_path),
    )

    model = load_training_model(_FakeTorch(), options, vocab_size=10)

    assert isinstance(model, _FakeTorch.nn.Module) and captured == {
        "vocab_size": 10,
        "hidden_dim": 8,
        "num_layers": 1,
        "attention_heads": 2,
        "mlp_hidden_dim": 64,
        "mlp_layers": 3,
        "dropout": 0.2,
        "position_embedding_type": "sinusoidal",
    }


def test_load_training_model_python_builder_supports_options_argument(tmp_path: Path) -> None:
    """Python architecture builder may accept options as a third argument."""
    architecture_path = tmp_path / "architecture.py"
    architecture_path.write_text(
        "\n".join(
            [
                "def build_model(vocab_size, torch_module, options):",
                "    _ = (vocab_size, options.hidden_dim)",
                "    return torch_module.nn.Module()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        architecture_path=str(architecture_path),
    )

    model = load_training_model(_FakeTorch(), options, vocab_size=10)

    assert isinstance(model, _FakeTorch.nn.Module)
