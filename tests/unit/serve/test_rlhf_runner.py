"""Unit tests for RLHF runner module."""

from __future__ import annotations

import pytest

from core.errors import CrucibleRlhfError
from core.rlhf_types import PpoConfig, RewardModelConfig, RlhfOptions


def test_rlhf_options_defaults() -> None:
    """RlhfOptions should have sensible defaults."""
    options = RlhfOptions(
        dataset_name="test",
        output_dir="/tmp/out",
        policy_model_path="/tmp/model.pt",
    )
    assert options.epochs == 3
    assert options.batch_size == 16
    assert options.learning_rate == 1e-5
    assert options.ppo_config.clip_epsilon == 0.2
    assert options.ppo_config.ppo_epochs == 4
    assert options.reward_config.reward_model_path is None
    assert options.reward_config.train_reward_model is False


def test_ppo_config_defaults() -> None:
    """PpoConfig should have standard PPO defaults."""
    config = PpoConfig()
    assert config.clip_epsilon == 0.2
    assert config.value_loss_coeff == 0.5
    assert config.entropy_coeff == 0.01
    assert config.ppo_epochs == 4
    assert config.gamma == 1.0
    assert config.lam == 0.95


def test_reward_model_config_external_path() -> None:
    """RewardModelConfig should accept external model path."""
    config = RewardModelConfig(reward_model_path="/tmp/reward.pt")
    assert config.reward_model_path == "/tmp/reward.pt"
    assert config.train_reward_model is False


def test_reward_model_config_train_from_preferences() -> None:
    """RewardModelConfig should support training from preference data."""
    config = RewardModelConfig(
        train_reward_model=True,
        preference_data_path="/tmp/prefs.jsonl",
    )
    assert config.train_reward_model is True
    assert config.preference_data_path == "/tmp/prefs.jsonl"


def test_rlhf_options_frozen() -> None:
    """RlhfOptions should be immutable."""
    options = RlhfOptions(
        dataset_name="test",
        output_dir="/tmp/out",
        policy_model_path="/tmp/model.pt",
    )
    with pytest.raises(AttributeError):
        options.epochs = 10  # type: ignore[misc]
