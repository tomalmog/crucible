"""Unit tests for RLHF CLI command wiring."""

from __future__ import annotations

import pytest

from cli.main import build_parser


def test_rlhf_command_registers_in_parser() -> None:
    """RLHF subcommand should be registered and parseable."""
    parser = build_parser()
    args = parser.parse_args([
        "rlhf-train",
        "--output-dir", "/tmp/out",
        "--policy-model-path", "/tmp/policy.pt",
    ])

    assert args.command == "rlhf-train"
    assert args.output_dir == "/tmp/out"
    assert args.policy_model_path == "/tmp/policy.pt"
    assert args.reward_model_path is None
    assert args.train_reward_model is False
    assert args.clip_epsilon == 0.2
    assert args.ppo_epochs == 4
    assert args.entropy_coeff == 0.01


def test_rlhf_command_requires_policy_model_path() -> None:
    """RLHF command should fail when --policy-model-path is missing."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "rlhf-train",
            "--output-dir", "/tmp/out",
        ])


def test_rlhf_command_accepts_reward_options() -> None:
    """RLHF command should accept reward model related arguments."""
    parser = build_parser()
    args = parser.parse_args([
        "rlhf-train",
        "--output-dir", "/tmp/out",
        "--policy-model-path", "/tmp/policy.pt",
        "--reward-model-path", "/tmp/reward.pt",
        "--train-reward-model",
        "--preference-data-path", "/tmp/prefs.jsonl",
        "--clip-epsilon", "0.3",
        "--ppo-epochs", "8",
        "--entropy-coeff", "0.02",
        "--epochs", "5",
        "--learning-rate", "1e-5",
    ])

    assert args.reward_model_path == "/tmp/reward.pt"
    assert args.train_reward_model is True
    assert args.preference_data_path == "/tmp/prefs.jsonl"
    assert args.clip_epsilon == 0.3
    assert args.ppo_epochs == 8
    assert args.entropy_coeff == 0.02
    assert args.epochs == 5
    assert args.learning_rate == 1e-5
