"""Unit tests for all 4 export format runners.

Tests cover SafeTensors, HF, ONNX, and GGUF export paths for both
Crucible .pt state-dict models and mocked HF classify_model results.
Dependencies that may not be installed (onnx, onnxruntime, gguf) are
checked at import time with pytest.importorskip.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Helpers: tiny model & state dict on disk
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Minimal nn.Module with an embedding and linear output layer."""

    def __init__(self, vocab: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.linear = nn.Linear(hidden, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.embedding(x))


def _save_pt_model(path: Path) -> str:
    """Save a tiny .pt file and return its string path."""
    model = _TinyModel()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))
    return str(path)


# ---------------------------------------------------------------------------
# SafeTensors export
# ---------------------------------------------------------------------------

class TestSafeTensorsExport:
    """SafeTensors export for Crucible .pt state dicts."""

    def test_export_crucible_pt(self, tmp_path: Path) -> None:
        """Exporting a .pt file produces a non-empty .safetensors file."""
        from core.safetensors_export_types import SafeTensorsExportOptions
        from serve.safetensors_exporter import run_safetensors_export

        pt_path = _save_pt_model(tmp_path / "model" / "model.pt")
        out_dir = tmp_path / "st_out"

        # classify_model should return (False, "") for a plain .pt with no
        # training_config.json indicating an HF base.
        result = run_safetensors_export(
            SafeTensorsExportOptions(model_path=pt_path, output_dir=str(out_dir)),
        )

        assert result["num_tensors"] > 0
        output = Path(result["output_path"])
        assert output.exists()
        assert output.stat().st_size > 0
        assert output.suffix == ".safetensors"

    def test_export_hf_model(self, tmp_path: Path) -> None:
        """Exporting an HF model ID produces safetensors via save_pretrained."""
        from core.safetensors_export_types import SafeTensorsExportOptions
        from serve.safetensors_exporter import run_safetensors_export

        out_dir = tmp_path / "st_hf_out"
        result = run_safetensors_export(
            SafeTensorsExportOptions(
                model_path="sshleifer/tiny-gpt2", output_dir=str(out_dir),
            ),
        )

        assert result["num_tensors"] > 0
        assert result["file_size_mb"] >= 0
        output = Path(result["output_path"])
        assert output.exists()
        assert output.stat().st_size > 0


# ---------------------------------------------------------------------------
# HuggingFace export
# ---------------------------------------------------------------------------

class TestHfExport:
    """HuggingFace-format export (safetensors + config.json)."""

    def test_export_crucible_pt(self, tmp_path: Path) -> None:
        """Exporting a .pt produces model.safetensors and config.json."""
        from core.hf_export_types import HfExportOptions
        from serve.hf_exporter import run_hf_export

        pt_path = _save_pt_model(tmp_path / "model" / "model.pt")
        out_dir = tmp_path / "hf_out"

        result = run_hf_export(
            HfExportOptions(model_path=pt_path, output_dir=str(out_dir)),
        )

        assert result["num_tensors"] > 0
        assert result["config_generated"] is True
        st_file = out_dir / "model.safetensors"
        assert st_file.exists() and st_file.stat().st_size > 0
        config_file = out_dir / "config.json"
        assert config_file.exists()
        config = json.loads(config_file.read_text())
        assert "model_type" in config

    def test_export_hf_model(self, tmp_path: Path) -> None:
        """Exporting an HF model ID produces a complete HF directory."""
        from core.hf_export_types import HfExportOptions
        from serve.hf_exporter import run_hf_export

        out_dir = tmp_path / "hf_hf_out"
        result = run_hf_export(
            HfExportOptions(
                model_path="sshleifer/tiny-gpt2", output_dir=str(out_dir),
            ),
        )

        assert result["num_tensors"] > 0
        assert result["tokenizer_copied"] is True
        # Should have config.json from save_pretrained
        assert (out_dir / "config.json").exists()


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

class TestOnnxExport:
    """ONNX export for Crucible .pt state dicts."""

    def test_export_crucible_pt(self, tmp_path: Path) -> None:
        """Exporting a Crucible .pt produces a valid .onnx file."""
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

        from core.onnx_export_types import OnnxExportOptions
        from serve.onnx_exporter import run_onnx_export

        # The onnx exporter for Crucible calls load_crucible_model which
        # goes through interp_model_loader. We mock that to return our tiny
        # model and a simple tokenizer.
        model = _TinyModel(vocab=32, hidden=16)
        mock_tok = MagicMock()

        pt_path = tmp_path / "model" / "model.pt"
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(pt_path))

        out_dir = tmp_path / "onnx_out"

        with patch("serve.interp_model_loader.load_interp_model", return_value=(model, mock_tok)):
            result = run_onnx_export(
                OnnxExportOptions(model_path=str(pt_path), output_dir=str(out_dir)),
            )

        onnx_file = Path(result["onnx_path"])
        assert onnx_file.exists()
        assert onnx_file.stat().st_size > 0
        assert result["file_size_mb"] >= 0
        assert "input_ids" in result["input_names"]


# ---------------------------------------------------------------------------
# GGUF export
# ---------------------------------------------------------------------------

class TestGgufExport:
    """GGUF export for Crucible .pt state dicts."""

    def test_export_crucible_pt_f32(self, tmp_path: Path) -> None:
        """Exporting a .pt file to GGUF F32 produces a valid .gguf file."""
        pytest.importorskip("gguf")

        from core.gguf_export_types import GgufExportOptions
        from serve.gguf_exporter import run_gguf_export

        pt_path = _save_pt_model(tmp_path / "model" / "model.pt")
        out_dir = tmp_path / "gguf_out"

        result = run_gguf_export(
            GgufExportOptions(
                model_path=pt_path, output_dir=str(out_dir), quant_type="F32",
            ),
        )

        gguf_file = Path(result["output_path"])
        assert gguf_file.exists()
        assert gguf_file.stat().st_size > 0
        assert result["num_tensors"] > 0
        assert result["quant_type"] == "F32"

    def test_export_crucible_pt_f16(self, tmp_path: Path) -> None:
        """Exporting a .pt file to GGUF F16 produces a valid .gguf file."""
        pytest.importorskip("gguf")

        from core.gguf_export_types import GgufExportOptions
        from serve.gguf_exporter import run_gguf_export

        pt_path = _save_pt_model(tmp_path / "model" / "model.pt")
        out_dir = tmp_path / "gguf_f16_out"

        result = run_gguf_export(
            GgufExportOptions(
                model_path=pt_path, output_dir=str(out_dir), quant_type="F16",
            ),
        )

        gguf_file = Path(result["output_path"])
        assert gguf_file.exists()
        assert result["quant_type"] == "F16"

    def test_quantized_type_raises(self, tmp_path: Path) -> None:
        """Requesting Q8_0 raises CrucibleDependencyError (needs llama.cpp)."""
        pytest.importorskip("gguf")

        from core.errors import CrucibleDependencyError
        from core.gguf_export_types import GgufExportOptions
        from serve.gguf_exporter import run_gguf_export

        pt_path = _save_pt_model(tmp_path / "model" / "model.pt")

        with pytest.raises(CrucibleDependencyError, match="llama.cpp"):
            run_gguf_export(
                GgufExportOptions(
                    model_path=pt_path,
                    output_dir=str(tmp_path / "q8"),
                    quant_type="Q8_0",
                ),
            )


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

class TestExportHelpers:
    """Unit tests for export_helpers functions."""

    def test_model_basename_pt(self) -> None:
        from serve.export_helpers import model_basename
        assert model_basename("/some/dir/mymodel.pt") == "mymodel"

    def test_model_basename_hf_id(self) -> None:
        from serve.export_helpers import model_basename
        assert model_basename("openai/gpt2") == "gpt2"

    def test_classify_model_hf(self) -> None:
        """classify_model identifies an HF model ID."""
        from serve.export_helpers import classify_model
        is_hf, model_id = classify_model("sshleifer/tiny-gpt2")
        assert is_hf is True
        assert model_id == "sshleifer/tiny-gpt2"

    def test_classify_model_pt(self, tmp_path: Path) -> None:
        """classify_model identifies a plain .pt file as non-HF."""
        from serve.export_helpers import classify_model
        pt_path = _save_pt_model(tmp_path / "m.pt")
        is_hf, _ = classify_model(pt_path)
        assert is_hf is False

    def test_load_state_dict_for_export(self, tmp_path: Path) -> None:
        """load_state_dict_for_export returns a dict of tensors."""
        from serve.export_helpers import load_state_dict_for_export
        pt_path = _save_pt_model(tmp_path / "model.pt")
        state = load_state_dict_for_export(pt_path)
        assert isinstance(state, dict)
        assert len(state) > 0
        for v in state.values():
            assert isinstance(v, torch.Tensor)

    def test_load_state_dict_missing_file(self) -> None:
        """load_state_dict_for_export raises on missing file."""
        from core.errors import CrucibleExportError
        from serve.export_helpers import load_state_dict_for_export
        with pytest.raises(CrucibleExportError, match="not found"):
            load_state_dict_for_export("/nonexistent/model.pt")
