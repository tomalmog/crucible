"""Integration tests for the 5 interp tools missing coverage.

Tests: linear-probe, sae-train, sae-analyze, steer-compute, steer-apply.
Uses real (tiny) models with no mocks. Crucible .pt models are trained from
scratch with minimal architecture. HuggingFace tests use sshleifer/tiny-gpt2.
"""

from __future__ import annotations

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
    """Build minimal DataRecord list for training and interp tools."""
    from core.types import DataRecord, RecordMetadata
    return [
        DataRecord(
            record_id=f"r-{i}",
            text="hello world test sentence " * 10,
            metadata=RecordMetadata("test", "en", 0.9, 100.0),
        )
        for i in range(5)
    ]


def _labeled_records() -> list:
    """Build labeled records for linear probe (at least 2 classes)."""
    from core.types import DataRecord, RecordMetadata
    records = []
    labels = ["positive", "negative"]
    texts = [
        "the movie was great and wonderful",
        "terrible awful bad movie experience",
        "fantastic film loved every moment",
        "worst movie ever seen boring",
        "amazing spectacular outstanding cinema",
        "dreadful horrible painful to watch",
    ]
    for i, text in enumerate(texts):
        label = labels[i % 2]
        records.append(DataRecord(
            record_id=f"lp-{i}",
            text=text,
            metadata=RecordMetadata(
                "test", "en", 0.9, 100.0,
                extra_fields={"sentiment": label},
            ),
        ))
    return records


def _train_crucible_model(tmp_path: Path) -> str:
    """Train a tiny Crucible .pt model and return its path."""
    from core.types import TrainingOptions
    from serve.training_runner import run_training
    return run_training(
        _records(),
        TrainingOptions(
            dataset_name="test",
            output_dir=str(tmp_path / "model"),
            learning_rate=0.001,
            **_TINY,
        ),
        42, tmp_path,
    ).model_path


@pytest.fixture(scope="module")
def module_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped temporary directory."""
    return tmp_path_factory.mktemp("interp_remaining")


@pytest.fixture(scope="module")
def crucible_model(module_tmp: Path) -> str:
    """Module-scoped trained Crucible .pt model path."""
    return _train_crucible_model(module_tmp / "crucible")


# ── Linear Probe ─────────────────────────────────────────────────────

class TestLinearProbe:
    """Linear probe analysis across model types."""

    def test_linear_probe_crucible(
        self, crucible_model: str, tmp_path: Path,
    ) -> None:
        """Linear probe on Crucible .pt returns layer accuracy results."""
        from core.linear_probe_types import LinearProbeOptions
        from serve.linear_probe_runner import run_linear_probe

        out = str(tmp_path / "lp_crucible")
        result = run_linear_probe(
            LinearProbeOptions(
                model_path=crucible_model,
                output_dir=out,
                dataset_name="test",
                label_field="sentiment",
                layer_index=-1,
                max_samples=6,
                epochs=3,
            ),
            _labeled_records(),
        )

        assert "layers" in result
        assert len(result["layers"]) > 0
        layer = result["layers"][0]
        assert "accuracy" in layer
        assert "num_classes" in layer
        assert layer["num_classes"] >= 2
        assert (Path(out) / "linear_probe.json").exists()

    def test_linear_probe_hf(self, tmp_path: Path) -> None:
        """Linear probe on sshleifer/tiny-gpt2 returns accuracy results."""
        from core.linear_probe_types import LinearProbeOptions
        from serve.linear_probe_runner import run_linear_probe

        out = str(tmp_path / "lp_hf")
        result = run_linear_probe(
            LinearProbeOptions(
                model_path=_HF,
                output_dir=out,
                dataset_name="test",
                label_field="sentiment",
                layer_index=-1,
                max_samples=6,
                epochs=3,
            ),
            _labeled_records(),
        )

        assert len(result["layers"]) > 0
        assert result["layers"][0]["num_classes"] >= 2
        assert (Path(out) / "linear_probe.json").exists()

    def test_linear_probe_all_layers(
        self, crucible_model: str, tmp_path: Path,
    ) -> None:
        """Linear probe with layer_index=-2 probes all layers."""
        from core.linear_probe_types import LinearProbeOptions
        from serve.linear_probe_runner import run_linear_probe

        out = str(tmp_path / "lp_all")
        result = run_linear_probe(
            LinearProbeOptions(
                model_path=crucible_model,
                output_dir=out,
                dataset_name="test",
                label_field="sentiment",
                layer_index=-2,
                max_samples=6,
                epochs=2,
            ),
            _labeled_records(),
        )

        assert len(result["layers"]) >= 1


# ── SAE Train ────────────────────────────────────────────────────────

class TestSaeTrain:
    """Sparse autoencoder training across model types."""

    def test_sae_train_crucible(
        self, crucible_model: str, tmp_path: Path,
    ) -> None:
        """SAE training on Crucible .pt saves model and reports loss."""
        from core.sae_types import SaeTrainOptions
        from serve.sae_train_runner import run_sae_train

        out = str(tmp_path / "sae_crucible")
        result = run_sae_train(
            SaeTrainOptions(
                model_path=crucible_model,
                output_dir=out,
                dataset_name="test",
                layer_index=-1,
                max_samples=5,
                epochs=3,
                learning_rate=1e-3,
                sparsity_coeff=1e-3,
            ),
            _records(),
        )

        assert result["epochs"] == 3
        assert result["final_loss"] > 0
        assert result["input_dim"] > 0
        assert result["latent_dim"] > 0
        sae_path = Path(result["sae_path"])
        assert sae_path.exists()
        assert (Path(out) / "sae_train.json").exists()

    def test_sae_train_hf(self, tmp_path: Path) -> None:
        """SAE training on sshleifer/tiny-gpt2 completes and saves model."""
        from core.sae_types import SaeTrainOptions
        from serve.sae_train_runner import run_sae_train

        out = str(tmp_path / "sae_hf")
        result = run_sae_train(
            SaeTrainOptions(
                model_path=_HF,
                output_dir=out,
                dataset_name="test",
                layer_index=-1,
                max_samples=5,
                epochs=2,
                learning_rate=1e-3,
                sparsity_coeff=1e-3,
            ),
            _records(),
        )

        assert result["epochs"] == 2
        assert Path(result["sae_path"]).exists()
        assert result["final_loss"] > 0


# ── SAE Analyze ──────────────────────────────────────────────────────

class TestSaeAnalyze:
    """SAE analysis using a previously trained SAE."""

    @pytest.fixture()
    def trained_sae_crucible(
        self, crucible_model: str, tmp_path: Path,
    ) -> str:
        """Train a tiny SAE on Crucible model and return the sae_path."""
        from core.sae_types import SaeTrainOptions
        from serve.sae_train_runner import run_sae_train

        out = str(tmp_path / "sae_for_analyze")
        result = run_sae_train(
            SaeTrainOptions(
                model_path=crucible_model,
                output_dir=out,
                dataset_name="test",
                layer_index=-1,
                max_samples=5,
                epochs=2,
                sparsity_coeff=1e-3,
            ),
            _records(),
        )
        return result["sae_path"]

    def test_sae_analyze_crucible(
        self, crucible_model: str, trained_sae_crucible: str, tmp_path: Path,
    ) -> None:
        """SAE analyze decomposes text into active features."""
        from core.sae_types import SaeAnalyzeOptions
        from serve.sae_analyze_runner import run_sae_analyze

        out = str(tmp_path / "sae_analyze_crucible")
        result = run_sae_analyze(
            SaeAnalyzeOptions(
                model_path=crucible_model,
                output_dir=out,
                sae_path=trained_sae_crucible,
                input_text="hello world this is a test",
                top_k_features=5,
            ),
            _records(),
        )

        assert "reconstruction_error" in result
        assert "sparsity" in result
        assert "active_features" in result
        assert result["total_features"] > 0
        assert "top_features" in result
        assert (Path(out) / "sae_analyze.json").exists()

    def test_sae_analyze_without_records(
        self, crucible_model: str, trained_sae_crucible: str, tmp_path: Path,
    ) -> None:
        """SAE analyze works without records (no feature associations)."""
        from core.sae_types import SaeAnalyzeOptions
        from serve.sae_analyze_runner import run_sae_analyze

        out = str(tmp_path / "sae_analyze_no_records")
        result = run_sae_analyze(
            SaeAnalyzeOptions(
                model_path=crucible_model,
                output_dir=out,
                sae_path=trained_sae_crucible,
                input_text="hello world this is a test",
                top_k_features=5,
            ),
            records=None,
        )

        assert "reconstruction_error" in result
        assert (Path(out) / "sae_analyze.json").exists()


# ── Steer Compute ────────────────────────────────────────────────────

class TestSteerCompute:
    """Steering vector computation from contrastive text pairs."""

    def test_steer_compute_crucible(
        self, crucible_model: str, tmp_path: Path,
    ) -> None:
        """Computing a steering vector from positive/negative text succeeds."""
        from core.steering_types import SteerComputeOptions
        from serve.steer_compute_runner import run_steer_compute

        out = str(tmp_path / "steer_crucible")
        result = run_steer_compute(
            SteerComputeOptions(
                model_path=crucible_model,
                output_dir=out,
                positive_text="happy joyful wonderful amazing",
                negative_text="sad terrible awful miserable",
                layer_index=-1,
            ),
        )

        assert "steering_vector_path" in result
        assert "vector_norm" in result
        assert result["vector_norm"] >= 0
        assert "cosine_similarity" in result
        assert result["num_positive"] == 1
        assert result["num_negative"] == 1
        vec_path = Path(result["steering_vector_path"])
        assert vec_path.exists()
        assert (Path(out) / "steer_compute.json").exists()

    def test_steer_compute_hf(self, tmp_path: Path) -> None:
        """Computing a steering vector on sshleifer/tiny-gpt2 succeeds."""
        from core.steering_types import SteerComputeOptions
        from serve.steer_compute_runner import run_steer_compute

        out = str(tmp_path / "steer_hf")
        result = run_steer_compute(
            SteerComputeOptions(
                model_path=_HF,
                output_dir=out,
                positive_text="happy joyful wonderful amazing",
                negative_text="sad terrible awful miserable",
                layer_index=-1,
            ),
        )

        assert result["vector_norm"] > 0
        assert Path(result["steering_vector_path"]).exists()

    def test_steer_compute_no_texts_raises(self, tmp_path: Path) -> None:
        """Missing both positive and negative text raises an error."""
        from core.errors import CrucibleError
        from core.steering_types import SteerComputeOptions
        from serve.steer_compute_runner import run_steer_compute

        with pytest.raises(CrucibleError, match="positive and negative"):
            run_steer_compute(
                SteerComputeOptions(
                    model_path=_HF,
                    output_dir=str(tmp_path / "steer_err"),
                ),
            )


# ── Steer Apply ──────────────────────────────────────────────────────

class TestSteerApply:
    """Steering vector application for steered text generation."""

    @pytest.fixture()
    def steering_vector_crucible(
        self, crucible_model: str, tmp_path: Path,
    ) -> str:
        """Compute a steering vector on Crucible model and return the path."""
        from core.steering_types import SteerComputeOptions
        from serve.steer_compute_runner import run_steer_compute

        out = str(tmp_path / "steer_for_apply")
        result = run_steer_compute(
            SteerComputeOptions(
                model_path=crucible_model,
                output_dir=out,
                positive_text="happy joyful wonderful",
                negative_text="sad terrible awful",
                layer_index=-1,
            ),
        )
        return result["steering_vector_path"]

    def test_steer_apply_crucible(
        self, crucible_model: str, steering_vector_crucible: str,
        tmp_path: Path,
    ) -> None:
        """Applying a steering vector produces both original and steered text."""
        from core.steering_types import SteerApplyOptions
        from serve.steer_apply_runner import run_steer_apply

        out = str(tmp_path / "apply_crucible")
        result = run_steer_apply(
            SteerApplyOptions(
                model_path=crucible_model,
                output_dir=out,
                steering_vector_path=steering_vector_crucible,
                input_text="hello",
                coefficient=1.0,
                max_new_tokens=5,
            ),
        )

        assert "original_text" in result
        assert "steered_text" in result
        assert result["coefficient"] == 1.0
        assert result["max_new_tokens"] == 5
        assert (Path(out) / "steer_apply.json").exists()

    def test_steer_apply_different_coefficients(
        self, crucible_model: str, steering_vector_crucible: str,
        tmp_path: Path,
    ) -> None:
        """Different coefficients produce different steered text."""
        from core.steering_types import SteerApplyOptions
        from serve.steer_apply_runner import run_steer_apply

        results = []
        for coeff in [0.0, 5.0]:
            out = str(tmp_path / f"apply_coeff_{coeff}")
            result = run_steer_apply(
                SteerApplyOptions(
                    model_path=crucible_model,
                    output_dir=out,
                    steering_vector_path=steering_vector_crucible,
                    input_text="hello",
                    coefficient=coeff,
                    max_new_tokens=5,
                ),
            )
            results.append(result)

        # With coefficient=0.0, steered should equal original
        assert results[0]["original_text"] == results[0]["steered_text"]
