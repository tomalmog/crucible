"""End-to-end integration tests for advanced training runners.

Covers grpo, qlora, kto, orpo, multimodal, and rlvr commands
(5 tests each, 30 total) with real PyTorch training — no mocks.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from cli.main import main
from core.grpo_types import GrpoOptions
from core.kto_types import KtoOptions
from core.multimodal_types import MultimodalOptions
from core.orpo_types import OrpoOptions
from core.qlora_types import QloraOptions
from core.rlvr_types import RlvrOptions
from core.types import DataRecord, RecordMetadata, TrainingRunResult
from core.types import TrainingOptions
from serve.grpo_runner import run_grpo_training
from serve.kto_runner import run_kto_training
from serve.multimodal_runner import run_multimodal_training
from serve.orpo_runner import run_orpo_training
from serve.qlora_runner import run_qlora_training
from serve.rlvr_runner import run_rlvr_training
from serve.training_runner import run_training

# ── Shared tiny config values ────────────────────────────────────────
_TINY = dict(
    hidden_dim=32,
    num_layers=1,
    attention_heads=2,
    batch_size=2,
    epochs=1,
    max_token_length=64,
    learning_rate=0.001,
    validation_split=0.2,
)


def _records() -> list[DataRecord]:
    """Five DataRecord objects with enough text for tokenizer fitting."""
    return [
        DataRecord(
            record_id=f"train-{i}",
            text="hello world test sentence " * 10,
            metadata=RecordMetadata("test", "en", 0.9, 100.0),
        )
        for i in range(5)
    ]


def _train_base_model(tmp_path: Path) -> str:
    """Train a tiny base model and return its model.pt path."""
    options = TrainingOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "base_model"),
        **_TINY,
    )
    result = run_training(_records(), options, 42, tmp_path, "v1")
    return result.model_path


def _assert_result(result: TrainingRunResult, epochs: int = 1) -> None:
    """Common assertions on a training run result."""
    assert Path(result.model_path).exists()
    assert result.epochs_completed == epochs
    assert result.run_id is not None
    assert result.history_path is not None
    assert Path(result.history_path).exists()


# ── GRPO ─────────────────────────────────────────────────────────────

def test_grpo_real_training(tmp_path: Path, grpo_data_file: str) -> None:
    options = GrpoOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "grpo_out"),
        grpo_data_path=grpo_data_file,
        **_TINY,
    )
    result = run_grpo_training(_records(), options, 42, tmp_path, "v1")
    _assert_result(result)


def test_grpo_cli_dispatches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ingested_dataset: tuple, grpo_data_file: str,
) -> None:
    client, ds_name, version_id = ingested_dataset
    code = main([
        "--data-root", str(client._config.data_root),
        "grpo-train",
        "--output-dir", str(tmp_path / "grpo_cli"),
        "--grpo-data-path", grpo_data_file,
        "--hidden-dim", "32", "--num-layers", "1",
        "--attention-heads", "2", "--batch-size", "2",
        "--epochs", "1", "--max-token-length", "64",
        "--learning-rate", "0.001",
    ])
    assert code == 0
    assert "model_path=" in capsys.readouterr().out


def test_grpo_cli_missing_dataset_errors(
    tmp_path: Path, grpo_data_file: str,
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["grpo-train", "--output-dir", str(tmp_path),
              "--grpo-data-path", grpo_data_file])
    assert exc.value.code == 2


def test_grpo_produces_artifacts(
    tmp_path: Path, grpo_data_file: str,
) -> None:
    options = GrpoOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "grpo_art"),
        grpo_data_path=grpo_data_file,
        **_TINY,
    )
    result = run_grpo_training(_records(), options, 42, tmp_path, "v1")
    assert Path(result.model_path).exists()
    assert Path(result.history_path).exists()
    config_path = Path(result.model_path).parent / "training_config.json"
    assert config_path.exists()


def test_grpo_loss_finite(tmp_path: Path, grpo_data_file: str) -> None:
    options = GrpoOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "grpo_loss"),
        grpo_data_path=grpo_data_file,
        **{**_TINY, "epochs": 2},
    )
    result = run_grpo_training(_records(), options, 42, tmp_path, "v1")
    assert result.epochs_completed == 2
    history = json.loads(Path(result.history_path).read_text())
    for entry in history["epochs"]:
        assert math.isfinite(entry["train_loss"])


# ── QLoRA ────────────────────────────────────────────────────────────

def test_qlora_real_training(
    tmp_path: Path, qlora_data_file: str,
) -> None:
    base_path = _train_base_model(tmp_path)
    options = QloraOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "qlora_out"),
        qlora_data_path=qlora_data_file,
        base_model_path=base_path,
        **_TINY,
    )
    result = run_qlora_training(_records(), options, 42, tmp_path, "v1")
    _assert_result(result)


def test_qlora_cli_dispatches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ingested_dataset: tuple, qlora_data_file: str,
) -> None:
    client, ds_name, version_id = ingested_dataset
    # Train base model on the SAME ingested dataset so vocab sizes match
    base_result = main([
        "--data-root", str(client._config.data_root),
        "train", "--dataset", ds_name,
        "--output-dir", str(tmp_path / "base_for_qlora"),
        "--hidden-dim", "32", "--num-layers", "1",
        "--attention-heads", "2", "--batch-size", "2",
        "--epochs", "1", "--max-token-length", "64",
        "--learning-rate", "0.001",
    ])
    captured = capsys.readouterr().out
    base_path = ""
    for line in captured.splitlines():
        if "model_path=" in line:
            base_path = line.split("model_path=")[1].strip()
            break
    code = main([
        "--data-root", str(client._config.data_root),
        "qlora-train",
        "--output-dir", str(tmp_path / "qlora_cli"),
        "--qlora-data-path", qlora_data_file,
        "--base-model-path", base_path,
        "--hidden-dim", "32", "--num-layers", "1",
        "--attention-heads", "2", "--batch-size", "2",
        "--epochs", "1", "--max-token-length", "64",
        "--learning-rate", "0.001",
    ])
    assert code == 0
    assert "model_path=" in capsys.readouterr().out


def test_qlora_cli_missing_dataset_errors(
    tmp_path: Path, qlora_data_file: str,
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["qlora-train", "--output-dir", str(tmp_path),
              "--qlora-data-path", qlora_data_file,
              "--base-model-path", str(tmp_path / "base.pt")])
    assert exc.value.code == 2


def test_qlora_produces_artifacts(
    tmp_path: Path, qlora_data_file: str,
) -> None:
    base_path = _train_base_model(tmp_path)
    options = QloraOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "qlora_art"),
        qlora_data_path=qlora_data_file,
        base_model_path=base_path,
        **_TINY,
    )
    result = run_qlora_training(_records(), options, 42, tmp_path, "v1")
    assert Path(result.model_path).exists()
    assert Path(result.history_path).exists()
    config_path = Path(result.model_path).parent / "training_config.json"
    assert config_path.exists()


def test_qlora_loss_finite(
    tmp_path: Path, qlora_data_file: str,
) -> None:
    base_path = _train_base_model(tmp_path)
    options = QloraOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "qlora_loss"),
        qlora_data_path=qlora_data_file,
        base_model_path=base_path,
        **{**_TINY, "epochs": 2},
    )
    result = run_qlora_training(_records(), options, 42, tmp_path, "v1")
    assert result.epochs_completed == 2
    history = json.loads(Path(result.history_path).read_text())
    for entry in history["epochs"]:
        assert math.isfinite(entry["train_loss"])


# ── KTO ──────────────────────────────────────────────────────────────

def test_kto_real_training(tmp_path: Path, kto_data_file: str) -> None:
    options = KtoOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "kto_out"),
        kto_data_path=kto_data_file,
        **_TINY,
    )
    result = run_kto_training(_records(), options, 42, tmp_path, "v1")
    _assert_result(result)


def test_kto_cli_dispatches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ingested_dataset: tuple, kto_data_file: str,
) -> None:
    client, ds_name, version_id = ingested_dataset
    code = main([
        "--data-root", str(client._config.data_root),
        "kto-train",
        "--output-dir", str(tmp_path / "kto_cli"),
        "--kto-data-path", kto_data_file,
        "--hidden-dim", "32", "--num-layers", "1",
        "--attention-heads", "2", "--batch-size", "2",
        "--epochs", "1", "--max-token-length", "64",
        "--learning-rate", "0.001",
    ])
    assert code == 0
    assert "model_path=" in capsys.readouterr().out


def test_kto_cli_missing_dataset_errors(
    tmp_path: Path, kto_data_file: str,
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["kto-train", "--output-dir", str(tmp_path),
              "--kto-data-path", kto_data_file])
    assert exc.value.code == 2


def test_kto_produces_artifacts(
    tmp_path: Path, kto_data_file: str,
) -> None:
    options = KtoOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "kto_art"),
        kto_data_path=kto_data_file,
        **_TINY,
    )
    result = run_kto_training(_records(), options, 42, tmp_path, "v1")
    assert Path(result.model_path).exists()
    assert Path(result.history_path).exists()
    config_path = Path(result.model_path).parent / "training_config.json"
    assert config_path.exists()


def test_kto_loss_finite(tmp_path: Path, kto_data_file: str) -> None:
    options = KtoOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "kto_loss"),
        kto_data_path=kto_data_file,
        **{**_TINY, "epochs": 2},
    )
    result = run_kto_training(_records(), options, 42, tmp_path, "v1")
    assert result.epochs_completed == 2
    history = json.loads(Path(result.history_path).read_text())
    for entry in history["epochs"]:
        assert math.isfinite(entry["train_loss"])


# ── ORPO ─────────────────────────────────────────────────────────────

def test_orpo_real_training(tmp_path: Path, orpo_data_file: str) -> None:
    options = OrpoOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "orpo_out"),
        orpo_data_path=orpo_data_file,
        **_TINY,
    )
    result = run_orpo_training(_records(), options, 42, tmp_path, "v1")
    _assert_result(result)


def test_orpo_cli_dispatches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ingested_dataset: tuple, orpo_data_file: str,
) -> None:
    client, ds_name, version_id = ingested_dataset
    code = main([
        "--data-root", str(client._config.data_root),
        "orpo-train",
        "--output-dir", str(tmp_path / "orpo_cli"),
        "--orpo-data-path", orpo_data_file,
        "--hidden-dim", "32", "--num-layers", "1",
        "--attention-heads", "2", "--batch-size", "2",
        "--epochs", "1", "--max-token-length", "64",
        "--learning-rate", "0.001",
    ])
    assert code == 0
    assert "model_path=" in capsys.readouterr().out


def test_orpo_cli_missing_dataset_errors(
    tmp_path: Path, orpo_data_file: str,
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["orpo-train", "--output-dir", str(tmp_path),
              "--orpo-data-path", orpo_data_file])
    assert exc.value.code == 2


def test_orpo_produces_artifacts(
    tmp_path: Path, orpo_data_file: str,
) -> None:
    options = OrpoOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "orpo_art"),
        orpo_data_path=orpo_data_file,
        **_TINY,
    )
    result = run_orpo_training(_records(), options, 42, tmp_path, "v1")
    assert Path(result.model_path).exists()
    assert Path(result.history_path).exists()
    config_path = Path(result.model_path).parent / "training_config.json"
    assert config_path.exists()


def test_orpo_loss_finite(tmp_path: Path, orpo_data_file: str) -> None:
    options = OrpoOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "orpo_loss"),
        orpo_data_path=orpo_data_file,
        **{**_TINY, "epochs": 2},
    )
    result = run_orpo_training(_records(), options, 42, tmp_path, "v1")
    assert result.epochs_completed == 2
    history = json.loads(Path(result.history_path).read_text())
    for entry in history["epochs"]:
        assert math.isfinite(entry["train_loss"])


# ── Multimodal ───────────────────────────────────────────────────────

def test_multimodal_real_training(
    tmp_path: Path, multimodal_data_file: str,
) -> None:
    options = MultimodalOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "mm_out"),
        multimodal_data_path=multimodal_data_file,
        **_TINY,
    )
    result = run_multimodal_training(_records(), options, 42, tmp_path, "v1")
    _assert_result(result)


def test_multimodal_cli_dispatches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ingested_dataset: tuple, multimodal_data_file: str,
) -> None:
    client, ds_name, version_id = ingested_dataset
    code = main([
        "--data-root", str(client._config.data_root),
        "multimodal-train",
        "--output-dir", str(tmp_path / "mm_cli"),
        "--multimodal-data-path", multimodal_data_file,
        "--hidden-dim", "32", "--num-layers", "1",
        "--attention-heads", "2", "--batch-size", "2",
        "--epochs", "1", "--max-token-length", "64",
        "--learning-rate", "0.001",
    ])
    assert code == 0
    assert "model_path=" in capsys.readouterr().out


def test_multimodal_cli_missing_dataset_errors(
    tmp_path: Path, multimodal_data_file: str,
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["multimodal-train", "--output-dir", str(tmp_path),
              "--multimodal-data-path", multimodal_data_file])
    assert exc.value.code == 2


def test_multimodal_produces_artifacts(
    tmp_path: Path, multimodal_data_file: str,
) -> None:
    options = MultimodalOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "mm_art"),
        multimodal_data_path=multimodal_data_file,
        **_TINY,
    )
    result = run_multimodal_training(_records(), options, 42, tmp_path, "v1")
    assert Path(result.model_path).exists()
    assert Path(result.history_path).exists()
    config_path = Path(result.model_path).parent / "training_config.json"
    assert config_path.exists()


def test_multimodal_loss_finite(
    tmp_path: Path, multimodal_data_file: str,
) -> None:
    options = MultimodalOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "mm_loss"),
        multimodal_data_path=multimodal_data_file,
        **{**_TINY, "epochs": 2},
    )
    result = run_multimodal_training(_records(), options, 42, tmp_path, "v1")
    assert result.epochs_completed == 2
    history = json.loads(Path(result.history_path).read_text())
    for entry in history["epochs"]:
        assert math.isfinite(entry["train_loss"])


# ── RLVR ─────────────────────────────────────────────────────────────

def test_rlvr_real_training(tmp_path: Path, rlvr_data_file: str) -> None:
    options = RlvrOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "rlvr_out"),
        rlvr_data_path=rlvr_data_file,
        **_TINY,
    )
    result = run_rlvr_training(_records(), options, 42, tmp_path, "v1")
    _assert_result(result)


def test_rlvr_cli_dispatches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ingested_dataset: tuple, rlvr_data_file: str,
) -> None:
    client, ds_name, version_id = ingested_dataset
    code = main([
        "--data-root", str(client._config.data_root),
        "rlvr-train",
        "--output-dir", str(tmp_path / "rlvr_cli"),
        "--rlvr-data-path", rlvr_data_file,
        "--hidden-dim", "32", "--num-layers", "1",
        "--attention-heads", "2", "--batch-size", "2",
        "--epochs", "1", "--max-token-length", "64",
        "--learning-rate", "0.001",
    ])
    assert code == 0
    assert "model_path=" in capsys.readouterr().out


def test_rlvr_cli_missing_dataset_errors(
    tmp_path: Path, rlvr_data_file: str,
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["rlvr-train", "--output-dir", str(tmp_path),
              "--rlvr-data-path", rlvr_data_file])
    assert exc.value.code == 2


def test_rlvr_produces_artifacts(
    tmp_path: Path, rlvr_data_file: str,
) -> None:
    options = RlvrOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "rlvr_art"),
        rlvr_data_path=rlvr_data_file,
        **_TINY,
    )
    result = run_rlvr_training(_records(), options, 42, tmp_path, "v1")
    assert Path(result.model_path).exists()
    assert Path(result.history_path).exists()
    config_path = Path(result.model_path).parent / "training_config.json"
    assert config_path.exists()


def test_rlvr_loss_finite(tmp_path: Path, rlvr_data_file: str) -> None:
    options = RlvrOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "rlvr_loss"),
        rlvr_data_path=rlvr_data_file,
        **{**_TINY, "epochs": 2},
    )
    result = run_rlvr_training(_records(), options, 42, tmp_path, "v1")
    assert result.epochs_completed == 2
    history = json.loads(Path(result.history_path).read_text())
    for entry in history["epochs"]:
        assert math.isfinite(entry["train_loss"])
