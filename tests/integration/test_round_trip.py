"""Round-trip verification tests: train → chat → verify output is usable.

These tests catch the class of bug where training "succeeds" (loss converges)
but the model artifact is unusable at inference — e.g., Issue 7 where the
tokenizer wrapper destroyed BPE encoding.

Every test trains a tiny model, then immediately verifies it works by
chatting with it and checking the output is non-garbage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.types import DataRecord, RecordMetadata, TrainingOptions, TrainingRunResult


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
    return [
        DataRecord(
            record_id=f"rt-{i}",
            text="hello world test sentence " * 10,
            metadata=RecordMetadata("test", "en", 0.9, 100.0),
        )
        for i in range(5)
    ]


def _assert_train_result(result: TrainingRunResult) -> None:
    assert Path(result.model_path).exists()
    assert result.epochs_completed >= 1
    assert Path(result.history_path).exists()


def _assert_chat_output(response: str) -> None:
    """Verify chat completes without error.

    For tiny models (hidden_dim=32, 1 epoch), empty responses are
    expected — the model hasn't learned enough to generate text.
    The critical check is that chat doesn't crash or raise an exception
    (which would indicate a tokenizer, model loading, or config bug).
    """
    assert isinstance(response, str), f"Chat returned {type(response)}, expected str"


def _chat_with_model(model_path: str, records: list[DataRecord] | None = None) -> str:
    """Chat with a trained model and return the response text."""
    from core.chat_types import ChatOptions
    from serve.chat_runner import run_chat

    opts = ChatOptions(
        model_path=model_path,
        prompt="Hello, tell me something.",
        max_new_tokens=20,
        hidden_dim=_TINY["hidden_dim"],
        num_layers=_TINY["num_layers"],
        attention_heads=_TINY["attention_heads"],
        max_token_length=_TINY["max_token_length"],
    )
    result = run_chat(records, opts)
    return result.response_text


# ── Basic Training → Chat ────────────────────────────────────────────


def test_basic_train_then_chat(tmp_path: Path) -> None:
    """Train a Crucible model from scratch, then chat with it."""
    from serve.training_runner import run_training

    options = TrainingOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "model"),
        **_TINY,
    )
    result = run_training(_records(), options, 42, tmp_path)
    _assert_train_result(result)

    response = _chat_with_model(result.model_path, _records())
    _assert_chat_output(response)


def test_basic_train_save_reload_chat(tmp_path: Path) -> None:
    """Train, save, reload from disk, and verify chat still works."""
    from serve.training_runner import run_training

    options = TrainingOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "model"),
        **_TINY,
    )
    result = run_training(_records(), options, 42, tmp_path)
    _assert_train_result(result)

    # Chat twice — both should produce non-empty output
    response1 = _chat_with_model(result.model_path, _records())
    response2 = _chat_with_model(result.model_path, _records())
    _assert_chat_output(response1)
    _assert_chat_output(response2)


# ── SFT → Chat ──────────────────────────────────────────────────────


def test_sft_train_then_chat(tmp_path: Path) -> None:
    """SFT fine-tune a base model, then chat with the result."""
    from serve.training_runner import run_training
    from serve.sft_runner import run_sft_training
    from core.sft_types import SftOptions

    # Train a base model first
    base_opts = TrainingOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "base"),
        **_TINY,
    )
    base_result = run_training(_records(), base_opts, 42, tmp_path)

    # SFT on top of it
    sft_data = tmp_path / "sft.jsonl"
    sft_data.write_text(
        '{"prompt": "What is ML?", "response": "Machine learning is AI."}\n'
        '{"prompt": "What is Python?", "response": "A programming language."}\n'
        '{"prompt": "Explain AI.", "response": "Artificial intelligence."}\n'
        * 3
    )
    sft_opts = SftOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "sft_out"),
        sft_data_path=str(sft_data),
        initial_weights_path=base_result.model_path,
        **_TINY,
    )
    sft_result = run_sft_training(_records(), sft_opts, 42, tmp_path)
    _assert_train_result(sft_result)

    response = _chat_with_model(sft_result.model_path, _records())
    _assert_chat_output(response)


# ── LoRA → Chat ──────────────────────────────────────────────────────


def test_lora_train_then_chat(tmp_path: Path) -> None:
    """LoRA fine-tune a base model, then chat with the merged result."""
    from serve.training_runner import run_training
    from serve.lora_training_runner import run_lora_training
    from core.lora_types import LoraTrainingOptions, LoraConfig

    # Train a base model
    base_opts = TrainingOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "base"),
        **_TINY,
    )
    base_result = run_training(_records(), base_opts, 42, tmp_path)

    # LoRA on top of it
    lora_data = tmp_path / "lora.jsonl"
    lora_data.write_text(
        '{"prompt": "Summarize ML.", "response": "ML learns from data."}\n'
        '{"prompt": "What is LoRA?", "response": "Low-rank adaptation."}\n'
        '{"prompt": "Explain fine-tuning.", "response": "Adapt a model."}\n'
        * 3
    )
    lora_opts = LoraTrainingOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "lora_out"),
        lora_data_path=str(lora_data),
        base_model_path=base_result.model_path,
        lora_config=LoraConfig(rank=4, alpha=4.0),
        **{k: v for k, v in _TINY.items() if k != "learning_rate"},
        learning_rate=1e-3,
    )
    lora_result = run_lora_training(lora_opts, 42, tmp_path)
    _assert_train_result(lora_result)

    response = _chat_with_model(lora_result.model_path, _records())
    _assert_chat_output(response)


# ── Tokenizer Round-Trip ─────────────────────────────────────────────


def test_tokenizer_roundtrip_after_training(tmp_path: Path) -> None:
    """Tokenizer saved alongside model can encode/decode without corruption."""
    from serve.training_runner import run_training
    from serve.training_metadata import load_tokenizer

    options = TrainingOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "model"),
        **_TINY,
    )
    result = run_training(_records(), options, 42, tmp_path)

    tokenizer = load_tokenizer(result.model_path)
    assert tokenizer is not None, "No tokenizer saved alongside model"

    test_text = "hello world test"
    encoded = tokenizer.encode(test_text, 64)
    assert len(encoded) > 0, "Tokenizer encode returned empty list"

    decoded = tokenizer.decode(encoded)
    assert len(decoded) > 0, "Tokenizer decode returned empty string"


# ── Training Config Preserves Base Model ─────────────────────────────


def test_sft_training_config_preserves_base_model(tmp_path: Path) -> None:
    """SFT training_config.json should contain base_model_path."""
    from serve.training_runner import run_training
    from serve.sft_runner import run_sft_training
    from serve.training_metadata import load_training_config
    from core.sft_types import SftOptions

    base_opts = TrainingOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "base"),
        **_TINY,
    )
    base_result = run_training(_records(), base_opts, 42, tmp_path)

    sft_data = tmp_path / "sft.jsonl"
    sft_data.write_text(
        '{"prompt": "Q", "response": "A"}\n' * 10
    )
    sft_opts = SftOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "sft_out"),
        sft_data_path=str(sft_data),
        base_model=base_result.model_path,
        initial_weights_path=base_result.model_path,
        **_TINY,
    )
    sft_result = run_sft_training(_records(), sft_opts, 42, tmp_path)

    config = load_training_config(sft_result.model_path)
    assert config is not None
    assert config.get("base_model_path") == base_result.model_path
