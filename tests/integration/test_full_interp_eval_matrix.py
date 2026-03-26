"""End-to-end tests for interpretability tools, benchmarks, and chat across model types.

Tests use real (tiny) models with no mocks. Crucible .pt models are trained from scratch
with minimal architecture. HuggingFace tests use sshleifer/tiny-gpt2.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_HF = "sshleifer/tiny-gpt2"
_TINY = dict(
    hidden_dim=32, num_layers=1, attention_heads=2,
    batch_size=2, epochs=1, max_token_length=64, validation_split=0.2,
)


def _records() -> list:
    """Build minimal DataRecord list for training and PCA."""
    from core.types import DataRecord, RecordMetadata
    return [
        DataRecord(record_id=f"r-{i}", text="hello world test sentence " * 10,
                   metadata=RecordMetadata("test", "en", 0.9, 100.0))
        for i in range(5)
    ]


def _train_crucible_model(tmp_path: Path) -> str:
    """Train a tiny Crucible .pt model and return its path."""
    from core.types import TrainingOptions
    from serve.training_runner import run_training
    return run_training(
        _records(),
        TrainingOptions(dataset_name="test", output_dir=str(tmp_path / "model"),
                        learning_rate=0.001, **_TINY),
        42, tmp_path,
    ).model_path


def _sft_data_file(tmp_path: Path) -> str:
    """Write a minimal SFT JSONL file and return its path."""
    path = tmp_path / "sft_data.jsonl"
    lines = [json.dumps({"prompt": f"Q{i}", "response": f"A{i}"}) for i in range(6)]
    path.write_text("\n".join(lines))
    return str(path)


@pytest.fixture(scope="module")
def module_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped temporary directory."""
    return tmp_path_factory.mktemp("interp_eval_matrix")


@pytest.fixture(scope="module")
def crucible_model(module_tmp: Path) -> str:
    """Module-scoped trained Crucible .pt model path."""
    return _train_crucible_model(module_tmp / "crucible")


# ── Interpretability: Logit Lens ─────────────────────────────────────

class TestLogitLens:
    """Logit lens analysis across model types."""

    def test_logit_lens_crucible(self, crucible_model: str, tmp_path: Path) -> None:
        """Logit lens on Crucible .pt returns layers with top tokens."""
        from core.logit_lens_types import LogitLensOptions
        from serve.logit_lens_runner import run_logit_lens
        out = str(tmp_path / "ll_crucible")
        result = run_logit_lens(LogitLensOptions(
            model_path=crucible_model, output_dir=out, input_text="hello world"))
        assert len(result["layers"]) > 0
        assert "predictions" in result["layers"][0]
        assert (Path(out) / "logit_lens.json").exists()

    def test_logit_lens_hf(self, tmp_path: Path) -> None:
        """Logit lens on sshleifer/tiny-gpt2 returns layer results."""
        from core.logit_lens_types import LogitLensOptions
        from serve.logit_lens_runner import run_logit_lens
        out = str(tmp_path / "ll_hf")
        result = run_logit_lens(LogitLensOptions(
            model_path=_HF, output_dir=out, input_text="the cat sat"))
        assert len(result["layers"]) > 0
        for layer in result["layers"]:
            for pred in layer["predictions"]:
                assert "top_k" in pred
        assert (Path(out) / "logit_lens.json").exists()


# ── Interpretability: Activation PCA ─────────────────────────────────

class TestActivationPca:
    """Activation PCA analysis across model types."""

    def test_pca_crucible(self, crucible_model: str, tmp_path: Path) -> None:
        """Activation PCA on Crucible .pt returns points with x,y."""
        from core.activation_pca_types import ActivationPcaOptions
        from serve.activation_pca_runner import run_activation_pca
        out = str(tmp_path / "pca_crucible")
        result = run_activation_pca(
            ActivationPcaOptions(model_path=crucible_model, output_dir=out,
                                 dataset_name="test", max_samples=5),
            _records())
        assert len(result["points"]) > 0
        assert "x" in result["points"][0] and "y" in result["points"][0]
        assert (Path(out) / "activation_pca.json").exists()

    def test_pca_hf(self, tmp_path: Path) -> None:
        """Activation PCA on sshleifer/tiny-gpt2 returns 2D points."""
        from core.activation_pca_types import ActivationPcaOptions
        from serve.activation_pca_runner import run_activation_pca
        out = str(tmp_path / "pca_hf")
        result = run_activation_pca(
            ActivationPcaOptions(model_path=_HF, output_dir=out,
                                 dataset_name="test", max_samples=5),
            _records())
        assert len(result["points"]) > 0
        for pt in result["points"]:
            assert "x" in pt and "y" in pt
        assert (Path(out) / "activation_pca.json").exists()


# ── Interpretability: Activation Patching ────────────────────────────

class TestActivationPatching:
    """Activation patching analysis across model types."""

    def test_patching_crucible(self, crucible_model: str, tmp_path: Path) -> None:
        """Activation patching on Crucible .pt returns layer effects."""
        from core.activation_patching_types import ActivationPatchingOptions
        from serve.activation_patching_runner import run_activation_patching
        out = str(tmp_path / "patch_crucible")
        result = run_activation_patching(ActivationPatchingOptions(
            model_path=crucible_model, output_dir=out,
            clean_text="hello world", corrupted_text="hello earth"))
        assert len(result["layer_results"]) > 0
        assert "recovery" in result["layer_results"][0]
        assert (Path(out) / "activation_patching.json").exists()

    def test_patching_hf(self, tmp_path: Path) -> None:
        """Activation patching on sshleifer/tiny-gpt2 returns layer effects."""
        from core.activation_patching_types import ActivationPatchingOptions
        from serve.activation_patching_runner import run_activation_patching
        out = str(tmp_path / "patch_hf")
        result = run_activation_patching(ActivationPatchingOptions(
            model_path=_HF, output_dir=out,
            clean_text="the cat sat", corrupted_text="the dog sat"))
        assert len(result["layer_results"]) > 0
        for lr in result["layer_results"]:
            assert "layer_name" in lr and "recovery" in lr
        assert (Path(out) / "activation_patching.json").exists()


# ── Eval / Benchmarks ────────────────────────────────────────────────

class TestEvalBenchmarks:
    """Benchmark runner across model types."""

    def test_eval_crucible_model(self, crucible_model: str, tmp_path: Path) -> None:
        """Run MMLU (max_samples=5) on Crucible .pt model."""
        from eval.benchmark_runner import run_benchmarks
        out = str(tmp_path / "eval_crucible.json")
        result = run_benchmarks(crucible_model, ["mmlu"], max_samples=5, output_path=out)
        assert len(result.benchmark_results) == 1
        assert result.benchmark_results[0].benchmark_name == "mmlu"
        assert result.benchmark_results[0].num_examples > 0
        assert Path(out).exists()

    def test_eval_hf_model(self, tmp_path: Path) -> None:
        """Run MMLU (max_samples=5) on sshleifer/tiny-gpt2."""
        from eval.benchmark_runner import run_benchmarks
        out = str(tmp_path / "eval_hf.json")
        result = run_benchmarks(_HF, ["mmlu"], max_samples=5, output_path=out)
        assert len(result.benchmark_results) == 1
        assert result.benchmark_results[0].num_examples > 0
        assert Path(out).exists()

    def test_eval_invalid_benchmark_raises(self, crucible_model: str) -> None:
        """Invalid benchmark name raises CrucibleBenchmarkError."""
        from core.errors import CrucibleBenchmarkError
        from eval.benchmark_runner import run_benchmarks
        with pytest.raises(CrucibleBenchmarkError):
            run_benchmarks(crucible_model, ["nonexistent_benchmark"])

    def test_eval_multiple_benchmarks(self, crucible_model: str, tmp_path: Path) -> None:
        """Run two benchmarks on Crucible model, verify both have results."""
        from eval.benchmark_runner import run_benchmarks
        out = str(tmp_path / "eval_multi.json")
        result = run_benchmarks(
            crucible_model, ["mmlu", "hellaswag"], max_samples=5, output_path=out)
        assert len(result.benchmark_results) == 2
        names = {r.benchmark_name for r in result.benchmark_results}
        assert names == {"mmlu", "hellaswag"}
        assert Path(out).exists()


# ── Chat ─────────────────────────────────────────────────────────────

class TestChat:
    """Chat inference across model types and training methods."""

    def test_chat_crucible(self, crucible_model: str) -> None:
        """Chat with Crucible .pt model does not crash."""
        from core.chat_types import ChatOptions
        from serve.chat_runner import run_chat
        result = run_chat(_records(), ChatOptions(
            model_path=crucible_model, prompt="hello",
            max_new_tokens=5, hidden_dim=32, num_layers=1,
            attention_heads=2, max_token_length=64))
        assert isinstance(result.response_text, str)

    def test_chat_hf(self) -> None:
        """Chat with sshleifer/tiny-gpt2 does not crash."""
        from core.chat_types import ChatOptions
        from serve.chat_runner import run_chat
        result = run_chat(None, ChatOptions(
            model_path=_HF, prompt="hello", max_new_tokens=5, temperature=0))
        assert isinstance(result.response_text, str)

    def test_chat_sft_trained_hf(self, module_tmp: Path) -> None:
        """SFT a tiny HF model, then chat with the result."""
        from core.sft_types import SftOptions
        from core.chat_types import ChatOptions
        from serve.sft_runner import run_sft_training
        from serve.chat_runner import run_chat
        sft_dir = module_tmp / "sft_hf"
        sft_dir.mkdir(exist_ok=True)
        data_path = _sft_data_file(sft_dir)
        train_result = run_sft_training(
            _records(),
            SftOptions(dataset_name="test", output_dir=str(sft_dir / "out"),
                       sft_data_path=data_path, base_model=_HF,
                       epochs=1, batch_size=2, max_token_length=64),
            random_seed=42, data_root=sft_dir)
        chat_result = run_chat(None, ChatOptions(
            model_path=train_result.model_path, prompt="hello",
            max_new_tokens=5, temperature=0))
        assert isinstance(chat_result.response_text, str)

    def test_chat_lora_trained_hf(self, module_tmp: Path) -> None:
        """LoRA fine-tune a tiny HF model, then chat with the result."""
        from core.lora_types import LoraConfig, LoraTrainingOptions
        from core.chat_types import ChatOptions
        from serve.lora_training_runner import run_lora_training
        from serve.chat_runner import run_chat
        lora_dir = module_tmp / "lora_hf"
        lora_dir.mkdir(exist_ok=True)
        data_path = _sft_data_file(lora_dir)
        train_result = run_lora_training(
            LoraTrainingOptions(
                dataset_name="test", output_dir=str(lora_dir / "out"),
                lora_data_path=data_path, base_model_path=_HF,
                lora_config=LoraConfig(rank=4, alpha=8.0),
                epochs=1, batch_size=2, max_token_length=64),
            random_seed=42, data_root=lora_dir)
        chat_result = run_chat(None, ChatOptions(
            model_path=train_result.model_path, prompt="hello",
            max_new_tokens=5, temperature=0))
        assert isinstance(chat_result.response_text, str)
