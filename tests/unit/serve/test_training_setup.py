"""Unit tests for training setup helpers."""

from __future__ import annotations

import pytest

from core.errors import CrucibleServeError
from core.types import DataRecord, RecordMetadata, TrainingOptions
from serve.training_setup import fit_training_tokenizer, validate_training_options


def _build_records() -> list[DataRecord]:
    metadata = RecordMetadata(
        source_uri="input.txt",
        language="en",
        quality_score=0.8,
        perplexity=2.1,
    )
    return [DataRecord(record_id="rid", text="alpha beta gamma", metadata=metadata)]


def test_validate_training_options_rejects_non_divisible_attention_shape(tmp_path) -> None:
    """Validation should fail when hidden dimension is not divisible by heads."""
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        hidden_dim=10,
        attention_heads=3,
    )

    with pytest.raises(CrucibleServeError):
        validate_training_options(options)

    assert True


def test_fit_training_tokenizer_applies_vocabulary_limit(tmp_path) -> None:
    """Tokenizer fitting should use configured vocabulary size cap."""
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        vocabulary_size=3,
    )
    tokenizer = fit_training_tokenizer(_build_records(), options)

    assert len(tokenizer.vocabulary) == 3


def test_validate_training_options_rejects_unknown_position_embedding(tmp_path) -> None:
    """Validation should fail for unsupported positional embedding type."""
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        position_embedding_type="unknown",  # type: ignore[arg-type]
    )

    with pytest.raises(CrucibleServeError):
        validate_training_options(options)

    assert True


def test_validate_training_options_rejects_non_positive_checkpoint_interval(tmp_path) -> None:
    """Validation should fail when checkpoint interval is below one epoch."""
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        checkpoint_every_epochs=0,
    )

    with pytest.raises(CrucibleServeError):
        validate_training_options(options)

    assert True


def test_validate_training_options_rejects_conflicting_weight_sources(tmp_path) -> None:
    """Validation should fail when initial and resume weights are both configured."""
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        initial_weights_path=str(tmp_path / "initial.pt"),
        resume_checkpoint_path=str(tmp_path / "checkpoint.pt"),
    )

    with pytest.raises(CrucibleServeError):
        validate_training_options(options)

    assert True


def test_validate_training_options_rejects_unknown_optimizer_type(tmp_path) -> None:
    """Validation should fail for unsupported optimizer type."""
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        optimizer_type="invalid",  # type: ignore[arg-type]
    )

    with pytest.raises(CrucibleServeError):
        validate_training_options(options)

    assert True


def test_validate_training_options_rejects_invalid_step_scheduler_gamma(tmp_path) -> None:
    """Validation should fail when step scheduler gamma is out of range."""
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        scheduler_type="step",
        scheduler_gamma=1.5,
    )

    with pytest.raises(CrucibleServeError):
        validate_training_options(options)

    assert True


def test_validate_training_options_rejects_unknown_precision_mode(tmp_path) -> None:
    """Validation should fail for unsupported mixed precision mode."""
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        precision_mode="mixed",  # type: ignore[arg-type]
    )

    with pytest.raises(CrucibleServeError):
        validate_training_options(options)

    assert True
