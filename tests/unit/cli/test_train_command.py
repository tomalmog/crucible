"""Unit tests for train CLI command wiring."""

from __future__ import annotations

from cli.main import main
from core.types import TrainingRunResult
from store.dataset_sdk import CrucibleClient


def test_cli_train_passes_checkpoint_options(monkeypatch, tmp_path, capsys) -> None:
    """Train command should forward checkpoint configuration flags."""
    captured: dict[str, object] = {}

    def _fake_train(self, options):
        captured["checkpoint_every_epochs"] = options.checkpoint_every_epochs
        captured["save_best_checkpoint"] = options.save_best_checkpoint
        captured["max_checkpoint_files"] = options.max_checkpoint_files
        captured["resume_checkpoint_path"] = options.resume_checkpoint_path
        captured["precision_mode"] = options.precision_mode
        captured["optimizer_type"] = options.optimizer_type
        captured["scheduler_type"] = options.scheduler_type
        captured["progress_log_interval_steps"] = options.progress_log_interval_steps
        return TrainingRunResult(
            model_path=str(tmp_path / "model.pt"),
            history_path=str(tmp_path / "history.json"),
            plot_path=None,
            epochs_completed=1,
        )

    monkeypatch.setattr(CrucibleClient, "train", _fake_train)
    resume_path = tmp_path / "epoch-0002.pt"
    exit_code = main(
        [
            "train",
            "--dataset",
            "demo",
            "--output-dir",
            str(tmp_path),
            "--checkpoint-every-epochs",
            "2",
            "--max-checkpoint-files",
            "9",
            "--no-save-best-checkpoint",
            "--resume-checkpoint-path",
            str(resume_path),
            "--precision-mode",
            "bf16",
            "--optimizer-type",
            "adamw",
            "--weight-decay",
            "0.01",
            "--scheduler-type",
            "cosine",
            "--scheduler-t-max-epochs",
            "12",
            "--progress-log-interval-steps",
            "3",
        ]
    )
    _ = capsys.readouterr()

    assert exit_code == 0 and captured == {
        "checkpoint_every_epochs": 2,
        "save_best_checkpoint": False,
        "max_checkpoint_files": 9,
        "resume_checkpoint_path": str(resume_path),
        "precision_mode": "bf16",
        "optimizer_type": "adamw",
        "scheduler_type": "cosine",
        "progress_log_interval_steps": 3,
    }
