"""Unit tests for replay CLI command wiring."""

from __future__ import annotations

import json

import pytest

from cli.main import build_parser, main
from core.types import TrainingRunResult
from store.dataset_sdk import ForgeClient


def test_replay_parser_requires_bundle_path() -> None:
    """Replay command should fail if --bundle-path is not provided."""
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["replay"])


def test_replay_parser_accepts_optional_output_dir() -> None:
    """Replay command should parse an optional --output-dir flag."""
    parser = build_parser()

    args = parser.parse_args([
        "replay",
        "--bundle-path", "/tmp/bundle.json",
        "--output-dir", "/tmp/replay_out",
    ])

    assert args.bundle_path == "/tmp/bundle.json"
    assert args.output_dir == "/tmp/replay_out"


def test_replay_command_invokes_train(monkeypatch, tmp_path) -> None:
    """Replay command should load the bundle and call client.train."""
    bundle_file = tmp_path / "reproducibility_bundle.json"
    bundle_data = {
        "run_id": "run-replay",
        "dataset_name": "demo",
        "dataset_version_id": "v1",
        "config_hash": "hash1",
        "random_seed": 42,
        "created_at": "2026-01-01T00:00:00+00:00",
        "python_version": "3.12.0",
        "platform": "linux",
        "training_options": {
            "dataset_name": "demo",
            "output_dir": str(tmp_path / "original_out"),
        },
    }
    bundle_file.write_text(json.dumps(bundle_data), encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_train(self, options):
        captured["dataset_name"] = options.dataset_name
        captured["output_dir"] = options.output_dir
        return TrainingRunResult(
            model_path=str(tmp_path / "model.pt"),
            history_path=str(tmp_path / "history.json"),
            plot_path=None,
            epochs_completed=1,
        )

    monkeypatch.setattr(ForgeClient, "train", _fake_train)
    override_dir = str(tmp_path / "replay_out")
    exit_code = main([
        "replay",
        "--bundle-path", str(bundle_file),
        "--output-dir", override_dir,
    ])

    assert exit_code == 0
    assert captured["dataset_name"] == "demo"
    assert captured["output_dir"] == override_dir
