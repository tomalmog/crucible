"""Shared fixtures for integration tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from core.config import ForgeConfig
from core.types import DataRecord, RecordMetadata, IngestOptions, TrainingRunResult
from store.dataset_sdk import ForgeClient


@pytest.fixture()
def forge_config(tmp_path: Path) -> ForgeConfig:
    """ForgeConfig rooted in a temporary directory."""
    return replace(ForgeConfig.from_env(), data_root=tmp_path)


@pytest.fixture()
def forge_client(forge_config: ForgeConfig) -> ForgeClient:
    """ForgeClient backed by the temporary data root."""
    return ForgeClient(forge_config)


@pytest.fixture()
def ingested_dataset(
    forge_client: ForgeClient, tmp_path: Path
) -> tuple[ForgeClient, str]:
    """Ingest two text files and return (client, dataset_name)."""
    raw_dir = tmp_path / "raw_files"
    raw_dir.mkdir()
    (raw_dir / "a.txt").write_text("Hello world this is a sample document.")
    (raw_dir / "b.txt").write_text("Another document with enough text content.")
    dataset_name = "test-ds"
    forge_client.ingest(
        IngestOptions(dataset_name=dataset_name, source_uri=str(raw_dir))
    )
    return forge_client, dataset_name


@pytest.fixture()
def sample_records() -> list[DataRecord]:
    """Three DataRecord objects for curator/distribution tests."""
    return [
        DataRecord(
            record_id="r1",
            text="This is a reasonably long document with enough text.",
            metadata=RecordMetadata(
                source_uri="file://a.txt",
                language="en",
                quality_score=0.9,
                perplexity=10.0,
            ),
        ),
        DataRecord(
            record_id="r2",
            text="Short",
            metadata=RecordMetadata(
                source_uri="file://b.txt",
                language="en",
                quality_score=0.5,
                perplexity=25.0,
            ),
        ),
        DataRecord(
            record_id="r3",
            text="Medium length text that has some content in it.",
            metadata=RecordMetadata(
                source_uri="file://c.txt",
                language="en",
                quality_score=0.7,
                perplexity=15.0,
            ),
        ),
    ]


@pytest.fixture()
def fake_training_result(tmp_path: Path) -> TrainingRunResult:
    """A TrainingRunResult with paths inside tmp_path."""
    return TrainingRunResult(
        model_path=str(tmp_path / "model.pt"),
        history_path=str(tmp_path / "history.json"),
        plot_path=str(tmp_path / "curves.png"),
        epochs_completed=3,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        best_checkpoint_path=str(tmp_path / "best.pt"),
        run_id="test-run-001",
        artifact_contract_path=str(tmp_path / "contract.json"),
    )


@pytest.fixture()
def grpo_data_file(tmp_path: Path) -> str:
    """JSONL file with prompt lines for GRPO training."""
    path = tmp_path / "grpo_data.jsonl"
    lines = [json.dumps({"prompt": f"Solve problem {i}"}) for i in range(5)]
    path.write_text("\n".join(lines))
    return str(path)


@pytest.fixture()
def kto_data_file(tmp_path: Path) -> str:
    """JSONL file with prompt/response/is_desirable lines for KTO."""
    path = tmp_path / "kto_data.jsonl"
    lines = [
        json.dumps({"prompt": "What is 2+2?", "response": "4", "is_desirable": True}),
        json.dumps(
            {"prompt": "What is 2+2?", "response": "5", "is_desirable": False}
        ),
        json.dumps(
            {"prompt": "Tell a joke", "response": "Why did the...", "is_desirable": True}
        ),
    ]
    path.write_text("\n".join(lines))
    return str(path)


@pytest.fixture()
def orpo_data_file(tmp_path: Path) -> str:
    """JSONL file with prompt/chosen/rejected for ORPO."""
    path = tmp_path / "orpo_data.jsonl"
    lines = [
        json.dumps(
            {"prompt": "Explain gravity", "chosen": "Gravity is...", "rejected": "Idk"}
        ),
        json.dumps(
            {
                "prompt": "What is Python?",
                "chosen": "A programming language",
                "rejected": "A snake",
            }
        ),
    ]
    path.write_text("\n".join(lines))
    return str(path)


@pytest.fixture()
def qlora_data_file(tmp_path: Path) -> str:
    """JSONL file with prompt/response pairs for QLoRA."""
    path = tmp_path / "qlora_data.jsonl"
    lines = [
        json.dumps({"prompt": f"Question {i}", "response": f"Answer for example {i}"})
        for i in range(5)
    ]
    path.write_text("\n".join(lines))
    return str(path)


@pytest.fixture()
def multimodal_data_file(tmp_path: Path) -> str:
    """JSONL file with text/image_path lines for multimodal."""
    path = tmp_path / "multimodal_data.jsonl"
    lines = [
        json.dumps({"text": f"Image caption {i}", "image_path": f"/img/{i}.jpg"})
        for i in range(3)
    ]
    path.write_text("\n".join(lines))
    return str(path)


@pytest.fixture()
def rlvr_data_file(tmp_path: Path) -> str:
    """JSONL file with prompt/solution lines for RLVR."""
    path = tmp_path / "rlvr_data.jsonl"
    lines = [
        json.dumps({"prompt": "def add(a, b):", "solution": "return a + b"}),
        json.dumps({"prompt": "def mul(a, b):", "solution": "return a * b"}),
    ]
    path.write_text("\n".join(lines))
    return str(path)


@pytest.fixture()
def training_records() -> list[DataRecord]:
    """Five DataRecord objects with enough text for tokenizer fitting."""
    return [
        DataRecord(
            record_id=f"train-{i}",
            text="hello world test sentence " * 10,
            metadata=RecordMetadata("test", "en", 0.9, 100.0),
        )
        for i in range(5)
    ]


@pytest.fixture()
def trained_model_path(tmp_path: Path, training_records: list[DataRecord]) -> str:
    """Train a tiny model and return its model.pt path (for merge tests)."""
    from core.types import TrainingOptions
    from serve.training_runner import run_training

    options = TrainingOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "base_model"),
        epochs=1,
        batch_size=2,
        hidden_dim=32,
        num_layers=1,
        attention_heads=2,
        max_token_length=64,
        learning_rate=0.001,
        validation_split=0.2,
    )
    result = run_training(
        training_records, options, 42, tmp_path,
    )
    return result.model_path


@pytest.fixture()
def seed_prompts_file(tmp_path: Path) -> str:
    """Plain text file with seed prompts for synthetic data generation."""
    path = tmp_path / "seed_prompts.txt"
    path.write_text(
        "Explain machine learning\n"
        "What is deep learning?\n"
        "How does backpropagation work?\n"
    )
    return str(path)


@pytest.fixture()
def test_prompts_file(tmp_path: Path) -> str:
    """JSONL file with prompt lines for judge evaluation."""
    path = tmp_path / "test_prompts.jsonl"
    lines = [
        json.dumps({"prompt": "What is AI?"}),
        json.dumps({"prompt": "Explain transformers."}),
    ]
    path.write_text("\n".join(lines))
    return str(path)
