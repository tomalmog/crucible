"""Unit tests for run-spec YAML export."""

from __future__ import annotations

import yaml

from core.run_spec_export import export_run_to_yaml, export_training_options_to_yaml
from core.types import TrainingOptions


def test_export_training_options_produces_valid_yaml() -> None:
    """Exported YAML should parse back into a valid structure."""
    options = TrainingOptions(
        dataset_name="test_ds",
        output_dir="/tmp/out",
        epochs=5,
        learning_rate=2e-4,
    )
    yaml_str = export_training_options_to_yaml(options)
    parsed = yaml.safe_load(yaml_str)

    assert parsed["version"] == 1
    assert parsed["defaults"]["dataset"] == "test_ds"
    assert len(parsed["steps"]) == 1
    assert parsed["steps"][0]["command"] == "train"
    assert parsed["steps"][0]["epochs"] == 5
    assert parsed["steps"][0]["learning_rate"] == 2e-4


def test_export_round_trip_preserves_options() -> None:
    """Options -> YAML -> parse should preserve all training parameters."""
    options = TrainingOptions(
        dataset_name="round_trip",
        output_dir="/tmp/rt",
        epochs=10,
        batch_size=32,
        precision_mode="bf16",
        optimizer_type="adamw",
    )
    yaml_str = export_training_options_to_yaml(options)
    parsed = yaml.safe_load(yaml_str)
    step = parsed["steps"][0]

    assert step["epochs"] == 10
    assert step["batch_size"] == 32
    assert step["precision_mode"] == "bf16"
    assert step["optimizer_type"] == "adamw"


def test_export_run_to_yaml_includes_run_id_comment() -> None:
    """When run_id is provided, output should include a header comment."""
    options = TrainingOptions(dataset_name="demo", output_dir="/tmp/d")
    yaml_str = export_run_to_yaml(options, run_id="run-abc-123")

    assert "run-abc-123" in yaml_str
    assert yaml_str.startswith("#")


def test_export_run_to_yaml_no_comment_without_run_id() -> None:
    """Without run_id, output should not start with a comment."""
    options = TrainingOptions(dataset_name="demo", output_dir="/tmp/d")
    yaml_str = export_run_to_yaml(options, run_id=None)

    assert not yaml_str.startswith("#")
    assert "version:" in yaml_str
