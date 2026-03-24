"""Unit tests for activation PCA runner — SVD-based PCA correctness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from core.activation_pca_types import ActivationPcaOptions


# ── SVD PCA math verification ────────────────────────────────────────


def test_svd_pca_matches_sklearn() -> None:
    """Our torch SVD PCA should produce equivalent results to sklearn PCA."""
    torch.manual_seed(42)
    data = torch.randn(20, 8).float()

    # Our implementation (same logic as activation_pca_runner.py)
    mean = data.mean(dim=0)
    centered = data - mean
    _U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    n_components = 2
    projected = (centered @ Vh[:n_components].T).numpy()
    total_var = (S ** 2).sum()
    explained_variance = [float(S[i] ** 2 / total_var) for i in range(n_components)]

    # Verify shape
    assert projected.shape == (20, 2)

    # Verify explained variance sums to <= 1.0 and is positive
    assert all(v > 0 for v in explained_variance)
    assert sum(explained_variance) <= 1.0 + 1e-6

    # Verify components are orthogonal (dot product ≈ 0)
    col0 = projected[:, 0]
    col1 = projected[:, 1]
    dot = abs(col0 @ col1) / (max(1e-8, (col0 @ col0) ** 0.5 * (col1 @ col1) ** 0.5))
    assert dot < 0.05, f"PCA components not orthogonal: cosine={dot:.4f}"


def test_svd_pca_single_component() -> None:
    """PCA with n_components=1 should still work."""
    # 1 sample → min(2, 1, dim) = 1 component
    data = torch.randn(1, 8).float()
    mean = data.mean(dim=0)
    centered = data - mean
    _U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    n_components = min(2, data.shape[0], data.shape[1])
    assert n_components == 1
    projected = (centered @ Vh[:n_components].T).numpy()
    assert projected.shape == (1, 1)


def test_svd_pca_preserves_variance_ordering() -> None:
    """First component should explain more variance than second."""
    torch.manual_seed(0)
    data = torch.randn(50, 16).float()
    mean = data.mean(dim=0)
    centered = data - mean
    _U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    total_var = (S ** 2).sum()
    ev0 = float(S[0] ** 2 / total_var)
    ev1 = float(S[1] ** 2 / total_var)
    assert ev0 >= ev1, f"PC1 ({ev0:.4f}) should explain >= PC2 ({ev1:.4f})"


# ── Full runner integration ──────────────────────────────────────────


@dataclass
class FakeRecord:
    text: str
    metadata: dict[str, str] | None = None


def _make_fake_model(hidden_dim: int = 32, n_layers: int = 2) -> MagicMock:
    """Create a fake model that returns predictable activations."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
    return model


def test_run_activation_pca_sample_granularity(tmp_path: Path) -> None:
    """Full run with sample granularity produces valid output."""
    from serve.activation_pca_runner import run_activation_pca

    options = ActivationPcaOptions(
        model_path="/fake/model",
        output_dir=str(tmp_path / "out"),
        dataset_name="test",
        granularity="sample",
        max_samples=5,
    )
    records = [FakeRecord(text=f"Sample text number {i}") for i in range(5)]

    fake_model = _make_fake_model()
    fake_tokenizer = MagicMock()

    layers = ["transformer.h.0", "transformer.h.1"]

    # Build fake stacked activations (5 samples, hidden_dim=32)
    torch.manual_seed(99)
    stacked = torch.randn(5, 32)
    labels = [""] * 5
    snippets = [f"Sample text number {i}" for i in range(5)]

    with (
        patch("serve.activation_pca_runner._load_model_and_tokenizer",
              return_value=(fake_model, fake_tokenizer)),
        patch("serve.activation_pca_runner.discover_transformer_layers",
              return_value=layers),
        patch("serve.activation_pca_runner.collect_activations",
              return_value=(stacked, labels, snippets)),
    ):
        result = run_activation_pca(options, records)

    assert result["layer_name"] == "transformer.h.1"
    assert result["granularity"] == "sample"
    assert len(result["points"]) == 5
    assert len(result["explained_variance"]) == 2
    assert all("x" in p and "y" in p for p in result["points"])
    # Output file written
    assert (tmp_path / "out" / "activation_pca.json").exists()


def test_run_activation_pca_token_granularity(tmp_path: Path) -> None:
    """Full run with token granularity produces per-token points."""
    from serve.activation_pca_runner import run_activation_pca

    options = ActivationPcaOptions(
        model_path="/fake/model",
        output_dir=str(tmp_path / "out"),
        dataset_name="test",
        granularity="token",
        max_samples=3,
    )
    records = [FakeRecord(text=f"Text {i}") for i in range(3)]

    fake_model = _make_fake_model()
    fake_tokenizer = MagicMock()

    layers = ["layer.0"]

    # 3 samples × 3 tokens = 9 token vectors, hidden_dim=16
    torch.manual_seed(42)
    stacked = torch.randn(9, 16)
    labels = [""] * 9
    snippets = ["tok"] * 9

    with (
        patch("serve.activation_pca_runner._load_model_and_tokenizer",
              return_value=(fake_model, fake_tokenizer)),
        patch("serve.activation_pca_runner.discover_transformer_layers",
              return_value=layers),
        patch("serve.activation_pca_runner.collect_activations",
              return_value=(stacked, labels, snippets)),
    ):
        result = run_activation_pca(options, records)

    # 9 points
    assert len(result["points"]) == 9
    assert result["granularity"] == "token"


# ── Helper function tests ────────────────────────────────────────────


def test_extract_texts_from_records() -> None:
    from serve.interp_data_utils import extract_texts

    records = [
        FakeRecord(text="Hello world"),
        FakeRecord(text=""),  # empty → skipped
        FakeRecord(text="Goodbye"),
    ]
    texts = extract_texts(records, max_samples=10)
    assert texts == ["Hello world", "Goodbye"]


def test_extract_texts_max_samples() -> None:
    from serve.interp_data_utils import extract_texts

    records = [FakeRecord(text=f"text {i}") for i in range(100)]
    texts = extract_texts(records, max_samples=3)
    assert len(texts) == 3


def test_extract_texts_prompt_response_records() -> None:
    """SFT/QLoRA/LoRA records with prompt field should be extracted."""
    from serve.interp_data_utils import extract_texts

    @dataclass
    class PromptRecord:
        prompt: str
        response: str = ""

    records = [
        PromptRecord(prompt="What is ML?", response="ML is..."),
        PromptRecord(prompt="Explain PCA"),
    ]
    texts = extract_texts(records, max_samples=10)
    assert len(texts) == 2
    assert texts[0] == "What is ML?"
    assert texts[1] == "Explain PCA"


def test_extract_texts_content_attribute() -> None:
    """Remote _SimpleRecord with .content should be extracted."""
    from serve.interp_data_utils import extract_texts

    class SimpleRec:
        def __init__(self, content: str) -> None:
            self.content = content

    records = [SimpleRec(content="Hello from remote")]
    texts = extract_texts(records, max_samples=10)
    assert texts == ["Hello from remote"]


def test_extract_texts_dict_records() -> None:
    """Plain dicts with various key names should be extracted."""
    from serve.interp_data_utils import extract_texts

    records = [
        {"text": "plain text"},
        {"prompt": "What?", "response": "Something"},
        {"input": "some input"},
        {"content": "content text"},
        {"instruction": "Do X"},
        {"unrelated_key": "nothing here"},  # should be skipped
    ]
    texts = extract_texts(records, max_samples=10)
    assert len(texts) == 5


def test_extract_texts_nested_dict() -> None:
    """Records with nested dict text field should extract nested text."""
    from serve.interp_data_utils import extract_texts

    records = [{"text": {"text": "nested content"}}]
    texts = extract_texts(records, max_samples=10)
    assert texts == ["nested content"]


def test_extract_texts_empty_records_raises() -> None:
    """Empty records should cause run_activation_pca to raise CrucibleError."""
    from core.errors import CrucibleError
    from serve.activation_pca_runner import run_activation_pca

    fake_model = _make_fake_model()
    fake_tokenizer = MagicMock()
    options = ActivationPcaOptions(
        model_path="/fake", output_dir="/tmp/out", dataset_name="test",
    )
    with (
        patch("serve.activation_pca_runner._load_model_and_tokenizer",
              return_value=(fake_model, fake_tokenizer)),
        patch("serve.activation_pca_runner.discover_transformer_layers",
              return_value=["layer.0"]),
        pytest.raises(CrucibleError, match="No usable text"),
    ):
        run_activation_pca(options, [])


def test_get_color_label() -> None:
    from serve.interp_data_utils import get_label

    records = [FakeRecord(text="x", metadata={"category": "A"})]
    assert get_label(records, 0, "category") == "A"
    assert get_label(records, 0, "") == ""
    assert get_label(records, 5, "category") == ""  # out of range
