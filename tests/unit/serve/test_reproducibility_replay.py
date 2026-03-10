"""Unit tests for reproducibility replay module."""

from __future__ import annotations

import json

import pytest

from core.errors import CrucibleServeError
from core.types import TrainingOptions
from serve.reproducibility_replay import (
    load_reproducibility_bundle,
    reconstruct_training_options,
)


def _valid_bundle_dict() -> dict[str, object]:
    """Return a minimal valid bundle dictionary."""
    return {
        "run_id": "run-abc",
        "dataset_name": "my_dataset",
        "config_hash": "abc123",
        "random_seed": 42,
        "created_at": "2026-01-01T00:00:00+00:00",
        "python_version": "3.12.0",
        "platform": "linux",
        "training_options": {
            "dataset_name": "my_dataset",
            "output_dir": "/tmp/out",
            "epochs": 5,
        },
    }


def test_load_valid_bundle(tmp_path) -> None:
    """Loading a well-formed bundle JSON file returns the parsed dict."""
    bundle_file = tmp_path / "reproducibility_bundle.json"
    payload = _valid_bundle_dict()
    bundle_file.write_text(json.dumps(payload), encoding="utf-8")

    result = load_reproducibility_bundle(str(bundle_file))

    assert result["run_id"] == "run-abc"
    assert result["dataset_name"] == "my_dataset"
    assert isinstance(result["training_options"], dict)


def test_load_missing_bundle(tmp_path) -> None:
    """Loading a non-existent path raises CrucibleServeError."""
    missing = tmp_path / "does_not_exist.json"

    with pytest.raises(CrucibleServeError, match="not found"):
        load_reproducibility_bundle(str(missing))


def test_load_invalid_json(tmp_path) -> None:
    """Loading a file with malformed JSON raises CrucibleServeError."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(CrucibleServeError, match="Invalid JSON"):
        load_reproducibility_bundle(str(bad_file))


def test_load_missing_training_options_key(tmp_path) -> None:
    """Loading a bundle without 'training_options' raises CrucibleServeError."""
    bundle_file = tmp_path / "bundle.json"
    bundle_file.write_text(json.dumps({"run_id": "run-1"}), encoding="utf-8")

    with pytest.raises(CrucibleServeError, match="missing"):
        load_reproducibility_bundle(str(bundle_file))


def test_reconstruct_training_options() -> None:
    """Valid training_options dict produces a TrainingOptions instance."""
    bundle = _valid_bundle_dict()

    options = reconstruct_training_options(bundle)

    assert isinstance(options, TrainingOptions)
    assert options.dataset_name == "my_dataset"
    assert options.output_dir == "/tmp/out"
    assert options.epochs == 5


def test_reconstruct_invalid_options() -> None:
    """An incompatible training_options dict raises CrucibleServeError."""
    bundle = _valid_bundle_dict()
    bundle["training_options"] = {"unknown_field": "bad_value"}

    with pytest.raises(CrucibleServeError, match="Cannot reconstruct"):
        reconstruct_training_options(bundle)
