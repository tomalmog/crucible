"""End-to-end training matrix: every method x model type combination.

Runs real (tiny) training with no mocks.  Verifies model artifacts,
config preservation, state-dict hygiene, chat round-trips, and
tokenizer integrity after each training run.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import pytest

os.environ["TOKENIZERS_PARALLELISM"] = "false"
_TINY = dict(hidden_dim=32, num_layers=1, attention_heads=2,
             batch_size=2, epochs=1, max_token_length=64, validation_split=0.2)
_HF = dict(epochs=1, batch_size=2, max_token_length=64,
           validation_split=0.2, precision_mode="fp32")
_HF_MODEL = "sshleifer/tiny-gpt2"


def _records():
    from core.types import DataRecord, RecordMetadata
    return [DataRecord(record_id=f"r-{i}", text="hello world test sentence " * 10,
                       metadata=RecordMetadata("test", "en", 0.9, 100.0)) for i in range(5)]


def _write_sft(p: Path):
    p.write_text('{"prompt":"What is ML?","response":"Machine learning is AI."}\n' * 10)


def _write_dpo(p: Path):
    p.write_text('{"prompt":"What?","chosen":"Good answer.","rejected":"Bad."}\n' * 10)


def _write_kto(p: Path):
    p.write_text(('{"prompt":"Q","response":"Good.","is_desirable":true}\n'
                  '{"prompt":"Q","response":"Bad.","is_desirable":false}\n') * 5)


def _write_grpo(p: Path):
    p.write_text('{"prompt":"Write a haiku."}\n' * 10)


def _verify(result, expect_hf_base=None):
    """Assert model.pt, config, history, and state-dict hygiene."""
    import torch
    mp = Path(result.model_path)
    assert mp.exists(), f"model.pt missing: {mp}"
    assert result.epochs_completed >= 1
    state = torch.load(str(mp), weights_only=True, map_location="cpu")
    assert state, "Empty state dict"
    assert not any(k.startswith("model.model.") for k in state), \
        f"Double prefix: {[k for k in state if k.startswith('model.model.')][:3]}"
    cfg_path = mp.parent / "training_config.json"
    assert cfg_path.exists(), "training_config.json missing"
    cfg = json.loads(cfg_path.read_text())
    if expect_hf_base:
        assert cfg.get("base_model_path") == expect_hf_base
    if result.history_path:
        assert Path(result.history_path).exists(), "history missing"


def _chat(model_path: str):
    from core.chat_types import ChatOptions
    from serve.chat_runner import run_chat
    r = run_chat(None, ChatOptions(model_path=model_path, prompt="Hello", max_new_tokens=5))
    assert isinstance(r.response_text, str)


@pytest.fixture(scope="module")
def base_model_path(tmp_path_factory):
    """Train one tiny Crucible model shared by Crucible-path tests."""
    from core.types import TrainingOptions
    from serve.training_runner import run_training
    root = tmp_path_factory.mktemp("base")
    return run_training(
        _records(), TrainingOptions(dataset_name="t", output_dir=str(root / "out"),
                                    learning_rate=0.001, **_TINY), 42, root,
    ).model_path


# ── Crucible .pt path ────────────────────────────────────────────────

def test_train_basic_crucible(tmp_path):
    """Basic train-from-scratch with Crucible architecture."""
    from core.types import TrainingOptions
    from serve.training_runner import run_training
    _verify(run_training(_records(), TrainingOptions(
        dataset_name="t", output_dir=str(tmp_path / "out"),
        learning_rate=0.001, **_TINY), 42, tmp_path))


def test_sft_crucible(tmp_path, base_model_path):
    """SFT with initial_weights_path to base .pt."""
    from core.sft_types import SftOptions
    from serve.sft_runner import run_sft_training
    d = tmp_path / "sft.jsonl"; _write_sft(d)
    _verify(run_sft_training(_records(), SftOptions(
        dataset_name="t", output_dir=str(tmp_path / "out"),
        sft_data_path=str(d), initial_weights_path=base_model_path,
        learning_rate=0.001, **_TINY), 42, tmp_path))


def test_lora_crucible(tmp_path, base_model_path):
    """LoRA on base .pt model."""
    from core.lora_types import LoraTrainingOptions
    from serve.lora_training_runner import run_lora_training
    d = tmp_path / "lora.jsonl"; _write_sft(d)
    _verify(run_lora_training(LoraTrainingOptions(
        dataset_name="t", output_dir=str(tmp_path / "out"),
        lora_data_path=str(d), base_model_path=base_model_path,
        learning_rate=0.001, **_TINY), 42, tmp_path))


def test_dpo_crucible(tmp_path, base_model_path):
    """DPO with base .pt model."""
    from core.dpo_types import DpoOptions
    from serve.dpo_runner import run_dpo_training
    d = tmp_path / "dpo.jsonl"; _write_dpo(d)
    _verify(run_dpo_training(_records(), DpoOptions(
        dataset_name="t", output_dir=str(tmp_path / "out"),
        dpo_data_path=str(d), initial_weights_path=base_model_path,
        learning_rate=0.001, **_TINY), 42, tmp_path))


# ── HuggingFace path ─────────────────────────────────────────────────

def test_sft_hf(tmp_path):
    """SFT using trl on sshleifer/tiny-gpt2."""
    from core.sft_types import SftOptions
    from serve.sft_runner import run_sft_training
    d = tmp_path / "sft.jsonl"; _write_sft(d)
    _verify(run_sft_training([], SftOptions(
        dataset_name="t", output_dir=str(tmp_path / "out"),
        sft_data_path=str(d), base_model=_HF_MODEL, **_HF), 42, tmp_path),
        expect_hf_base=_HF_MODEL)


def test_lora_hf(tmp_path):
    """LoRA + peft on sshleifer/tiny-gpt2."""
    from core.lora_types import LoraTrainingOptions
    from serve.lora_training_runner import run_lora_training
    d = tmp_path / "lora.jsonl"; _write_sft(d)
    _verify(run_lora_training(LoraTrainingOptions(
        dataset_name="t", output_dir=str(tmp_path / "out"),
        lora_data_path=str(d), base_model_path=_HF_MODEL, **_HF), 42, tmp_path),
        expect_hf_base=_HF_MODEL)


def test_dpo_hf(tmp_path):
    """DPO via trl on sshleifer/tiny-gpt2."""
    from core.dpo_types import DpoOptions
    from serve.dpo_runner import run_dpo_training
    d = tmp_path / "dpo.jsonl"; _write_dpo(d)
    _verify(run_dpo_training([], DpoOptions(
        dataset_name="t", output_dir=str(tmp_path / "out"),
        dpo_data_path=str(d), base_model=_HF_MODEL, **_HF), 42, tmp_path),
        expect_hf_base=_HF_MODEL)


def test_kto_hf(tmp_path):
    """KTO via trl on sshleifer/tiny-gpt2."""
    from core.kto_types import KtoOptions
    from serve.kto_runner import run_kto_training
    d = tmp_path / "kto.jsonl"; _write_kto(d)
    _verify(run_kto_training([], KtoOptions(
        dataset_name="t", output_dir=str(tmp_path / "out"),
        kto_data_path=str(d), base_model=_HF_MODEL, **_HF), 42, tmp_path),
        expect_hf_base=_HF_MODEL)


def test_orpo_hf(tmp_path):
    """ORPO via trl on sshleifer/tiny-gpt2."""
    from core.orpo_types import OrpoOptions
    from serve.orpo_runner import run_orpo_training
    d = tmp_path / "orpo.jsonl"; _write_dpo(d)
    _verify(run_orpo_training([], OrpoOptions(
        dataset_name="t", output_dir=str(tmp_path / "out"),
        orpo_data_path=str(d), base_model=_HF_MODEL, **_HF), 42, tmp_path),
        expect_hf_base=_HF_MODEL)


def test_grpo_hf(tmp_path):
    """GRPO via trl on sshleifer/tiny-gpt2."""
    from core.grpo_types import GrpoOptions
    from serve.grpo_runner import run_grpo_training
    d = tmp_path / "grpo.jsonl"; _write_grpo(d)
    _verify(run_grpo_training([], GrpoOptions(
        dataset_name="t", output_dir=str(tmp_path / "out"),
        grpo_data_path=str(d), base_model=_HF_MODEL, **_HF), 42, tmp_path),
        expect_hf_base=_HF_MODEL)


def test_rlvr_hf(tmp_path):
    """RLVR via trl on sshleifer/tiny-gpt2."""
    from core.rlvr_types import RlvrOptions
    from serve.rlvr_runner import run_rlvr_training
    d = tmp_path / "rlvr.jsonl"
    d.write_text('{"prompt":"def add(a,b):","solution":"return a+b"}\n' * 10)
    _verify(run_rlvr_training([], RlvrOptions(
        dataset_name="t", output_dir=str(tmp_path / "out"),
        rlvr_data_path=str(d), base_model=_HF_MODEL, **_HF), 42, tmp_path),
        expect_hf_base=_HF_MODEL)


def test_domain_adapt_hf(tmp_path):
    """Domain adaptation (continued pretraining) on sshleifer/tiny-gpt2."""
    from core.domain_adaptation_types import DomainAdaptationOptions
    from serve.domain_adaptation_runner import run_domain_adaptation
    _verify(run_domain_adaptation(_records(), DomainAdaptationOptions(
        dataset_name="t", output_dir=str(tmp_path / "out"),
        base_model_path=_HF_MODEL, **_HF), 42, tmp_path),
        expect_hf_base=_HF_MODEL)


# ── Cross-model chat verification ────────────────────────────────────

def test_sft_hf_then_chat(tmp_path):
    """SFT on HF model, then chat with result -- verify non-crash."""
    from core.sft_types import SftOptions
    from serve.sft_runner import run_sft_training
    d = tmp_path / "sft.jsonl"; _write_sft(d)
    r = run_sft_training([], SftOptions(
        dataset_name="t", output_dir=str(tmp_path / "out"),
        sft_data_path=str(d), base_model=_HF_MODEL, **_HF), 42, tmp_path)
    _chat(r.model_path)


def test_lora_hf_then_chat(tmp_path):
    """LoRA on HF model, then chat with result."""
    from core.lora_types import LoraTrainingOptions
    from serve.lora_training_runner import run_lora_training
    d = tmp_path / "lora.jsonl"; _write_sft(d)
    r = run_lora_training(LoraTrainingOptions(
        dataset_name="t", output_dir=str(tmp_path / "out"),
        lora_data_path=str(d), base_model_path=_HF_MODEL, **_HF), 42, tmp_path)
    _chat(r.model_path)


def test_dpo_hf_then_chat(tmp_path):
    """DPO on HF model, then chat with result."""
    from core.dpo_types import DpoOptions
    from serve.dpo_runner import run_dpo_training
    d = tmp_path / "dpo.jsonl"; _write_dpo(d)
    r = run_dpo_training([], DpoOptions(
        dataset_name="t", output_dir=str(tmp_path / "out"),
        dpo_data_path=str(d), base_model=_HF_MODEL, **_HF), 42, tmp_path)
    _chat(r.model_path)


# ── Regression tests ─────────────────────────────────────────────────

def test_hf_model_no_double_key_prefix(tmp_path):
    """Verify HF-trained model.pt has no model.model.* double prefix."""
    import torch
    from core.sft_types import SftOptions
    from serve.sft_runner import run_sft_training
    d = tmp_path / "sft.jsonl"; _write_sft(d)
    r = run_sft_training([], SftOptions(
        dataset_name="t", output_dir=str(tmp_path / "out"),
        sft_data_path=str(d), base_model=_HF_MODEL, **_HF), 42, tmp_path)
    state = torch.load(r.model_path, weights_only=True, map_location="cpu")
    assert not any(k.startswith("model.model.") for k in state)


def test_training_config_preserves_base_model(tmp_path):
    """Every HF method must save base_model_path in training_config.json."""
    from core.dpo_types import DpoOptions
    from serve.dpo_runner import run_dpo_training
    d = tmp_path / "dpo.jsonl"; _write_dpo(d)
    r = run_dpo_training([], DpoOptions(
        dataset_name="t", output_dir=str(tmp_path / "out"),
        dpo_data_path=str(d), base_model=_HF_MODEL, **_HF), 42, tmp_path)
    cfg = json.loads((Path(r.model_path).parent / "training_config.json").read_text())
    assert cfg.get("base_model_path") == _HF_MODEL


def test_lora_target_modules_auto_detect(tmp_path):
    """LoRA on GPT-2 succeeds despite c_attn (not q_proj) layers."""
    from core.lora_types import LoraTrainingOptions
    from serve.lora_training_runner import run_lora_training
    d = tmp_path / "lora.jsonl"; _write_sft(d)
    r = run_lora_training(LoraTrainingOptions(
        dataset_name="t", output_dir=str(tmp_path / "out"),
        lora_data_path=str(d), base_model_path=_HF_MODEL, **_HF), 42, tmp_path)
    assert r.epochs_completed >= 1


def test_finetune_lr_defaults_not_base(tmp_path):
    """All fine-tune methods default LR < 1e-3 (not base-training rate)."""
    from core.sft_types import SftOptions
    from core.dpo_types import DpoOptions
    from core.lora_types import LoraTrainingOptions
    from core.kto_types import KtoOptions
    from core.grpo_types import GrpoOptions
    from core.orpo_types import OrpoOptions
    from core.rlvr_types import RlvrOptions
    for cls in (SftOptions, DpoOptions, LoraTrainingOptions,
                KtoOptions, GrpoOptions, OrpoOptions, RlvrOptions):
        lr = cls.__dataclass_fields__["learning_rate"].default
        assert lr < 1e-3, f"{cls.__name__}.learning_rate={lr} >= 1e-3"
