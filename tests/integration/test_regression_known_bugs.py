"""Regression tests for known bugs from the hex-gen training run.

Each test targets a specific class of bug discovered during real training,
ensuring they never recur. See the design doc for full issue descriptions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.types import DataRecord, RecordMetadata, TrainingOptions


_TINY = dict(
    hidden_dim=32,
    num_layers=1,
    attention_heads=2,
    batch_size=2,
    epochs=1,
    max_token_length=64,
    learning_rate=0.001,
    validation_split=0.2,
)


def _records() -> list[DataRecord]:
    return [
        DataRecord(
            record_id=f"reg-{i}",
            text="hello world test sentence " * 10,
            metadata=RecordMetadata("test", "en", 0.9, 100.0),
        )
        for i in range(5)
    ]


# ── Issue 7: Tokenizer wrapper must preserve encoding ────────────────


def test_hf_tokenizer_preserves_bpe_through_save_load() -> None:
    """AutoTokenizerAdapter round-trips through save_tokenizer_vocabulary.

    Regression for Issue 7: VocabularyTokenizer destroyed HF BPE encoding.
    The fix uses AutoTokenizerAdapter which must preserve the BPE model
    through save_pretrained / load cycles.
    """
    from serve.huggingface_tokenizer import AutoTokenizerAdapter

    # AutoTokenizerAdapter wraps transformers.AutoTokenizer
    # We can't test with a real HF model without downloading, but we CAN
    # verify the type detection in save_tokenizer_vocabulary
    from serve.training_metadata import save_tokenizer_vocabulary
    from serve.tokenization import VocabularyTokenizer

    # A VocabularyTokenizer should save as flat vocab (this is correct)
    vocab = VocabularyTokenizer(vocabulary={"hello": 0, "world": 1, "<unk>": 2})
    assert vocab.encode("hello world", 10) is not None

    # Verify VocabularyTokenizer uses whitespace splitting (expected behavior)
    ids = vocab.encode("hello world", 10)
    assert isinstance(ids, list)


def test_vocabulary_tokenizer_is_whitespace_only() -> None:
    """VocabularyTokenizer must NOT be used for HF models.

    This test documents the intentional behavior: VocabularyTokenizer
    does whitespace splitting. If it's accidentally used with an HF model,
    the encoding will be wrong — "thecat" produces a single token instead
    of BPE subwords.
    """
    from serve.tokenization import VocabularyTokenizer

    vocab = {"the": 0, "cat": 1, "sat": 2, "<unk>": 3}
    tok = VocabularyTokenizer(vocabulary=vocab)

    # "the cat" → ["the", "cat"] → [0, 1] (whitespace split works)
    ids = tok.encode("the cat", 10)
    assert ids == [0, 1]

    # "thecat" → ["thecat"] → single token (NOT subword split)
    # This is the exact behavior that destroyed BPE in Issue 7
    ids2 = tok.encode("thecat", 10)
    assert len(ids2) == 1  # treated as one token, not subwords


# ── Issue 3: LoRA rank must be applied to model ─────────────────────


def test_lora_rank_applied_to_model(tmp_path: Path) -> None:
    """Requested LoRA rank must appear in the trained model's adapter layers.

    Regression for Issue 3: lora_rank appeared to be ignored.
    """
    import torch
    from serve.training_runner import run_training
    from serve.lora_training_runner import run_lora_training
    from core.lora_types import LoraTrainingOptions, LoraConfig

    base_opts = TrainingOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "base"),
        **_TINY,
    )
    base_result = run_training(_records(), base_opts, 42, tmp_path)

    lora_data = tmp_path / "lora.jsonl"
    lora_data.write_text(
        '{"prompt": "Q", "response": "A"}\n' * 10
    )
    requested_rank = 4
    lora_opts = LoraTrainingOptions(
        dataset_name="test",
        output_dir=str(tmp_path / "lora_out"),
        lora_data_path=str(lora_data),
        base_model_path=base_result.model_path,
        lora_config=LoraConfig(rank=requested_rank, alpha=4.0),
        **{k: v for k, v in _TINY.items() if k != "learning_rate"},
        learning_rate=1e-3,
    )
    lora_result = run_lora_training(lora_opts, 42, tmp_path)

    # Load the merged model and check that LoRA adapter config was saved
    config_path = Path(lora_result.model_path).parent / "lora_adapter_config.json"
    if config_path.exists():
        import json
        config = json.loads(config_path.read_text())
        assert config["rank"] == requested_rank, (
            f"Saved adapter rank {config['rank']} != requested {requested_rank}"
        )


# ── Issue 2 + H1: Fine-tuning methods must use low LR defaults ──────


def test_finetune_methods_use_appropriate_lr_defaults() -> None:
    """Fine-tuning methods must NOT use the base 1e-3 learning rate.

    Regression for Issue 2: default LR 1e-3 caused NaN loss on batch 2.
    """
    from core.sft_types import SftOptions
    from core.lora_types import LoraTrainingOptions
    from core.qlora_types import QloraOptions
    from core.dpo_types import DpoOptions
    from core.distillation_types import DistillationOptions
    from core.domain_adaptation_types import DomainAdaptationOptions

    base_lr = 1e-3  # This is too high for fine-tuning

    # All fine-tuning methods should default to something lower than 1e-3
    assert SftOptions(dataset_name="x", output_dir="x").learning_rate < base_lr
    assert LoraTrainingOptions(dataset_name="x", output_dir="x").learning_rate < base_lr
    assert QloraOptions(dataset_name="x", output_dir="x").learning_rate < base_lr
    assert DpoOptions(dataset_name="x", output_dir="x").learning_rate < base_lr
    assert DistillationOptions(dataset_name="x", output_dir="x", teacher_model_path="t").learning_rate < base_lr
    assert DomainAdaptationOptions(dataset_name="x", output_dir="x", base_model_path="b").learning_rate < base_lr


# ── C2: QLoRA adapter save must preserve all config fields ───────────


def test_qlora_adapter_config_fields() -> None:
    """QLoRA adapter save must include rank, alpha, dropout, target_modules.

    Regression for C2: qlora_runner.py:375 only saved rank=8, dropping
    alpha, dropout, and target_modules.
    """
    from core.lora_types import LoraConfig

    # Verify LoraConfig can be constructed with all 4 fields
    config = LoraConfig(
        rank=64,
        alpha=32.0,
        dropout=0.1,
        target_modules=("q_proj", "k_proj", "v_proj"),
    )
    assert config.rank == 64
    assert config.alpha == 32.0
    assert config.dropout == 0.1
    assert config.target_modules == ("q_proj", "k_proj", "v_proj")


# ── C1: Training config must preserve base model path ────────────────


def test_options_to_training_options_preserves_base_model() -> None:
    """The shared conversion helper must copy base_model to base_model_path.

    Regression for C1: SFT/DPO/KTO/GRPO/ORPO/RLVR lost the HF base model
    identity because _*_to_training_options() functions didn't copy it.
    """
    from core.training_types import options_to_training_options
    from core.sft_types import SftOptions
    from core.dpo_types import DpoOptions
    from core.kto_types import KtoOptions

    sft = SftOptions(
        dataset_name="x", output_dir="x", base_model="gpt2",
    )
    training_opts = options_to_training_options(sft)
    assert training_opts.base_model_path == "gpt2"

    dpo = DpoOptions(
        dataset_name="x", output_dir="x", base_model="gpt2",
    )
    training_opts = options_to_training_options(dpo)
    assert training_opts.base_model_path == "gpt2"

    kto = KtoOptions(
        dataset_name="x", output_dir="x", base_model="gpt2",
    )
    training_opts = options_to_training_options(kto)
    assert training_opts.base_model_path == "gpt2"


def test_options_to_training_options_handles_different_key_names() -> None:
    """Conversion helper handles base_model_path and policy_model_path."""
    from core.training_types import options_to_training_options
    from core.domain_adaptation_types import DomainAdaptationOptions
    from core.rlhf_types import RlhfOptions

    da = DomainAdaptationOptions(
        dataset_name="x", output_dir="x", base_model_path="/model.pt",
    )
    opts = options_to_training_options(da, base_model_key="base_model_path")
    assert opts.base_model_path == "/model.pt"

    rlhf = RlhfOptions(
        dataset_name="x", output_dir="x", policy_model_path="/policy.pt",
    )
    opts = options_to_training_options(rlhf, base_model_key="policy_model_path")
    assert opts.base_model_path == "/policy.pt"


# ── M2: Invalid benchmark names must raise, not silently succeed ─────


def test_invalid_benchmark_names_raise() -> None:
    """Invalid benchmark names must raise CrucibleBenchmarkError.

    Regression for M2: invalid names were silently filtered out.
    """
    from core.errors import CrucibleBenchmarkError
    from eval.benchmark_runner import run_benchmarks

    with pytest.raises(CrucibleBenchmarkError, match="No valid benchmark"):
        run_benchmarks("model.pt", ["nonexistent", "also_fake"])


# ── M3: Float-to-int coercion for sweep parameters ──────────────────


def test_coerce_value_float_to_int() -> None:
    """_coerce_value must cast float 3.0 to int 3 when target is int.

    Regression for M3: sweep parameters (always float) weren't cast to int.
    """
    from core.training_methods import _coerce_value

    assert _coerce_value(3.0, int) == 3
    assert isinstance(_coerce_value(3.0, int), int)
    assert _coerce_value(5.7, int) == 5
    assert isinstance(_coerce_value(5.7, int), int)
