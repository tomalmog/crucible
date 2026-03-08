"""Unit tests for run-spec parsing."""

from __future__ import annotations

import pytest

from core.errors import ForgeRunSpecError
from core.run_spec import load_run_spec
from tests.fixture_paths import fixture_path


def test_load_run_spec_valid_pipeline_parses_steps() -> None:
    """Valid run-spec should parse expected command order."""
    spec = load_run_spec(str(fixture_path("run_spec/valid_pipeline.yaml")))
    assert tuple(step.command for step in spec.steps) == ("ingest",)


def test_load_run_spec_invalid_command_raises_error() -> None:
    """Unsupported command name should raise run-spec error."""
    with pytest.raises(ForgeRunSpecError):
        load_run_spec(str(fixture_path("run_spec/invalid_command.yaml")))
    assert True


def test_load_run_spec_invalid_defaults_key_raises_error() -> None:
    """Unknown defaults field should be rejected."""
    with pytest.raises(ForgeRunSpecError):
        load_run_spec(str(fixture_path("run_spec/invalid_defaults_key.yaml")))
    assert True


def test_load_run_spec_hardware_profile_command_parses_without_dataset() -> None:
    """Hardware-profile run-spec step should parse as a supported command."""
    spec = load_run_spec(str(fixture_path("run_spec/hardware_profile.yaml")))

    assert tuple(step.command for step in spec.steps) == ("hardware-profile",)
