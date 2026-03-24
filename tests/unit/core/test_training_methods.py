"""Unit tests for training method dispatch and type coercion."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from core.lora_types import LoraConfig, LoraTrainingOptions
from core.qlora_types import QloraOptions
from core.rlhf_types import PpoConfig, RewardModelConfig, RlhfOptions
from core.sft_types import SftOptions
from core.training_methods import (
    _coerce_dataclass_kwargs,
    _coerce_value,
    _nest_flat_keys,
    dispatch_training,
)


# -- _coerce_value ----------------------------------------------------------

def test_coerce_str_to_int() -> None:
    assert _coerce_value("64", int) == 64


def test_coerce_str_float_to_int() -> None:
    assert _coerce_value("3.0", int) == 3


def test_coerce_str_to_float() -> None:
    assert _coerce_value("0.001", float) == pytest.approx(0.001)


def test_coerce_str_to_bool_true() -> None:
    assert _coerce_value("true", bool) is True


def test_coerce_str_to_bool_false() -> None:
    assert _coerce_value("false", bool) is False


def test_coerce_str_to_bool_one() -> None:
    assert _coerce_value("1", bool) is True


def test_coerce_str_to_bool_zero() -> None:
    assert _coerce_value("0", bool) is False


def test_coerce_str_to_tuple() -> None:
    result = _coerce_value("q_proj,v_proj", tuple[str, ...])
    assert result == ("q_proj", "v_proj")


def test_coerce_str_to_tuple_strips_whitespace() -> None:
    result = _coerce_value("q_proj , v_proj", tuple[str, ...])
    assert result == ("q_proj", "v_proj")


def test_coerce_list_to_tuple() -> None:
    result = _coerce_value(["a", "b"], tuple[str, ...])
    assert result == ("a", "b")


def test_coerce_dict_to_dataclass() -> None:
    result = _coerce_value({"rank": "32", "alpha": "16.0"}, LoraConfig)
    assert isinstance(result, LoraConfig)
    assert result.rank == 32
    assert result.alpha == 16.0


def test_coerce_optional_unwrap() -> None:
    assert _coerce_value("42", int | None) == 42


def test_coerce_none_passthrough() -> None:
    assert _coerce_value(None, int) is None


def test_coerce_already_correct_type() -> None:
    assert _coerce_value(64, int) == 64


# -- _coerce_dataclass_kwargs -----------------------------------------------

def test_coerce_sft_options_from_strings() -> None:
    opts = _coerce_dataclass_kwargs(SftOptions, {
        "dataset_name": "ds",
        "output_dir": "/out",
        "epochs": "5",
        "learning_rate": "0.001",
        "mask_prompt_tokens": "true",
    })
    assert isinstance(opts, SftOptions)
    assert opts.epochs == 5
    assert opts.learning_rate == pytest.approx(0.001)
    assert opts.mask_prompt_tokens is True


def test_coerce_lora_options_with_nested_dict() -> None:
    opts = _coerce_dataclass_kwargs(LoraTrainingOptions, {
        "dataset_name": "ds",
        "output_dir": "/out",
        "lora_config": {"rank": "64", "alpha": "32.0"},
    })
    assert isinstance(opts, LoraTrainingOptions)
    assert opts.lora_config.rank == 64
    assert opts.lora_config.alpha == 32.0


def test_coerce_ignores_unknown_keys() -> None:
    opts = _coerce_dataclass_kwargs(SftOptions, {
        "dataset_name": "ds",
        "output_dir": "/out",
        "totally_fake_key": "ignored",
    })
    assert opts.dataset_name == "ds"


def test_coerce_optional_none() -> None:
    opts = _coerce_dataclass_kwargs(SftOptions, {
        "dataset_name": "ds",
        "output_dir": "/out",
        "hooks_path": None,
    })
    assert opts.hooks_path is None


# -- _nest_flat_keys ---------------------------------------------------------

def test_nest_lora_flat_keys() -> None:
    result = _nest_flat_keys("lora-train", {
        "dataset_name": "ds",
        "lora_rank": "64",
        "lora_alpha": "32",
        "lora_dropout": "0.1",
        "lora_target_modules": "q_proj,v_proj",
    })
    assert result["dataset_name"] == "ds"
    assert result["lora_config"] == {
        "rank": "64",
        "alpha": "32",
        "dropout": "0.1",
        "target_modules": "q_proj,v_proj",
    }
    assert "lora_rank" not in result


def test_nest_rlhf_flat_keys() -> None:
    result = _nest_flat_keys("rlhf-train", {
        "policy_model_path": "/model",
        "reward_model_path": "/reward",
        "clip_epsilon": "0.3",
        "ppo_epochs": "8",
    })
    assert result["policy_model_path"] == "/model"
    assert result["reward_config"] == {"reward_model_path": "/reward"}
    assert result["ppo_config"] == {"clip_epsilon": "0.3", "ppo_epochs": "8"}


def test_nest_passthrough_for_unknown_method() -> None:
    original = {"dataset_name": "ds", "epochs": "5"}
    result = _nest_flat_keys("sft", original)
    assert result == original


# -- dispatch_training round-trips -------------------------------------------

def test_dispatch_lora_with_flat_string_keys() -> None:
    mock_client = MagicMock()
    mock_client.lora_train.return_value = "result"
    dispatch_training(mock_client, "lora-train", {
        "dataset_name": "ds",
        "output_dir": "/out",
        "lora_rank": "64",
        "lora_alpha": "32",
        "epochs": "10",
    })
    call_args = mock_client.lora_train.call_args
    opts: LoraTrainingOptions = call_args[0][0]
    assert opts.lora_config.rank == 64
    assert opts.lora_config.alpha == 32.0
    assert opts.epochs == 10


def test_dispatch_lora_with_nested_dict() -> None:
    mock_client = MagicMock()
    mock_client.lora_train.return_value = "result"
    dispatch_training(mock_client, "lora-train", {
        "dataset_name": "ds",
        "output_dir": "/out",
        "lora_config": {"rank": 64, "alpha": 32},
    })
    opts: LoraTrainingOptions = mock_client.lora_train.call_args[0][0]
    assert opts.lora_config.rank == 64


def test_dispatch_rlhf_with_flat_keys() -> None:
    mock_client = MagicMock()
    mock_client.rlhf_train.return_value = "result"
    dispatch_training(mock_client, "rlhf-train", {
        "dataset_name": "ds",
        "output_dir": "/out",
        "policy_model_path": "/model",
        "reward_model_path": "/reward",
        "clip_epsilon": "0.3",
        "ppo_epochs": "8",
    })
    opts: RlhfOptions = mock_client.rlhf_train.call_args[0][0]
    assert opts.reward_config.reward_model_path == "/reward"
    assert opts.ppo_config.clip_epsilon == pytest.approx(0.3)
    assert opts.ppo_config.ppo_epochs == 8


def test_dispatch_qlora_string_tuple() -> None:
    mock_client = MagicMock()
    mock_client.qlora_train.return_value = "result"
    dispatch_training(mock_client, "qlora-train", {
        "dataset_name": "ds",
        "output_dir": "/out",
        "lora_target_modules": "q_proj,v_proj",
        "lora_rank": "32",
    })
    opts: QloraOptions = mock_client.qlora_train.call_args[0][0]
    assert opts.lora_target_modules == ("q_proj", "v_proj")
    assert opts.lora_rank == 32


def test_dispatch_sft_string_coercion() -> None:
    mock_client = MagicMock()
    mock_client.sft_train.return_value = "result"
    dispatch_training(mock_client, "sft", {
        "dataset_name": "ds",
        "output_dir": "/out",
        "epochs": "5",
        "learning_rate": "0.0003",
        "batch_size": "16",
    })
    opts: SftOptions = mock_client.sft_train.call_args[0][0]
    assert opts.epochs == 5
    assert opts.learning_rate == pytest.approx(0.0003)
    assert opts.batch_size == 16


def test_dispatch_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown training method"):
        dispatch_training(MagicMock(), "fake-method", {})
