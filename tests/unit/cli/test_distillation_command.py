"""Unit tests for distillation CLI command wiring."""

from __future__ import annotations

import pytest

from cli.main import build_parser


def test_distill_command_registers_in_parser() -> None:
    """Distill subcommand should be registered and parseable."""
    parser = build_parser()
    args = parser.parse_args([
        "distill",
        "--dataset", "demo",
        "--output-dir", "/tmp/out",
        "--teacher-model-path", "/tmp/teacher.pt",
    ])

    assert args.command == "distill"
    assert args.dataset == "demo"
    assert args.output_dir == "/tmp/out"
    assert args.teacher_model_path == "/tmp/teacher.pt"
    assert args.student_model_path is None
    assert args.temperature == 2.0
    assert args.alpha == 0.5


def test_distill_command_accepts_optional_args() -> None:
    """Distill subcommand should accept optional training arguments."""
    parser = build_parser()
    args = parser.parse_args([
        "distill",
        "--dataset", "demo",
        "--output-dir", "/tmp/out",
        "--teacher-model-path", "/tmp/teacher.pt",
        "--student-model-path", "/tmp/student.pt",
        "--temperature", "4.0",
        "--alpha", "0.7",
        "--epochs", "5",
        "--learning-rate", "0.01",
        "--batch-size", "32",
    ])

    assert args.student_model_path == "/tmp/student.pt"
    assert args.temperature == 4.0
    assert args.alpha == 0.7
    assert args.epochs == 5
    assert args.learning_rate == 0.01
    assert args.batch_size == 32


def test_distill_command_requires_teacher_model_path() -> None:
    """Distill command should fail when --teacher-model-path is missing."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "distill",
            "--dataset", "demo",
            "--output-dir", "/tmp/out",
        ])


def test_distill_command_requires_dataset() -> None:
    """Distill command should fail when --dataset is missing."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "distill",
            "--output-dir", "/tmp/out",
            "--teacher-model-path", "/tmp/teacher.pt",
        ])


def test_distill_command_requires_output_dir() -> None:
    """Distill command should fail when --output-dir is missing."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "distill",
            "--dataset", "demo",
            "--teacher-model-path", "/tmp/teacher.pt",
        ])
