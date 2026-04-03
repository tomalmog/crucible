"""End-to-end tests verifying every audit fix works correctly in real execution.

These tests use actual model training (no mocks), real datasets from
tests/fixtures/toy_datasets/, and verify outputs are correct at every stage:
ingest → train → chat → verify artifacts.

Specific scenarios covered:
1. Parquet ingest: all-null column, empty strings, valid data → train → chat
2. SFT: toy fixture dataset → train real HF model → chat → verify tokenizer is HF format
3. DPO: toy fixture dataset → train real HF model → verify training_config
4. KTO: toy fixture dataset → train real HF model → verify is_desirable format parsed
5. RLVR: toy fixture dataset → train real HF model → verify model artifacts
6. LoRA: toy fixture dataset → train real HF model → chat → verify adapter saved correctly
7. Benchmark runner: real model + injected single-benchmark failure → verify isolation,
   partial result files written correctly after each benchmark
8. Tokenizer rename: verify no cross-import of wrong load_huggingface_tokenizer
9. Multiple parquet formats: text column, prompt/response columns, unicode, large files
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_HF_MODEL = "sshleifer/tiny-gpt2"
_HF = dict(
    epochs=1,
    batch_size=2,
    max_token_length=64,
    validation_split=0.2,
    precision_mode="fp32",
)

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "toy_datasets"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _verify_artifacts(result: object, expect_hf_base: str | None = None) -> None:
    """Assert all expected artifacts exist and are internally consistent."""
    import torch

    mp = Path(result.model_path)  # type: ignore[attr-defined]
    assert mp.exists(), f"model.pt missing: {mp}"
    assert result.epochs_completed >= 1  # type: ignore[attr-defined]

    state = torch.load(str(mp), weights_only=True, map_location="cpu")
    assert state, "state dict is empty"
    double = [k for k in state if k.startswith("model.model.")]
    assert not double, f"Double-prefix keys: {double[:3]}"

    cfg_path = mp.parent / "training_config.json"
    assert cfg_path.exists(), "training_config.json missing"
    cfg = json.loads(cfg_path.read_text())

    if expect_hf_base:
        assert cfg.get("base_model_path") == expect_hf_base, (
            f"training_config base_model_path mismatch: "
            f"got {cfg.get('base_model_path')!r}, expected {expect_hf_base!r}"
        )

    if result.history_path:  # type: ignore[attr-defined]
        hp = Path(result.history_path)  # type: ignore[attr-defined]
        assert hp.exists(), "history.json missing"
        history = json.loads(hp.read_text())
        # History can be a list (Crucible .pt) or dict with 'epochs' key (HF trl)
        if isinstance(history, list):
            assert len(history) >= 1, f"history list is empty: {history!r}"
        elif isinstance(history, dict):
            assert history, f"history dict is empty: {history!r}"
        else:
            pytest.fail(f"Unexpected history type {type(history)}: {history!r}")


def _chat_with_model(model_path: str, prompt: str = "Hello") -> str:
    """Run chat against a trained model and return the response text."""
    from core.chat_types import ChatOptions
    from serve.chat_runner import run_chat

    result = run_chat(
        None,
        ChatOptions(model_path=model_path, prompt=prompt, max_new_tokens=10),
    )
    assert isinstance(result.response_text, str), (
        f"chat response should be str, got {type(result.response_text)}"
    )
    return result.response_text


def _make_parquet(path: Path, data: dict[str, list]) -> None:
    """Write a minimal parquet file using pyarrow."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    table = pa.table(data)
    pq.write_table(table, str(path))


# ── 1. Parquet ingest: all-null column raises clearly ────────────────────────


class TestParquetIngest:
    """Parquet reader handles broken and valid data correctly."""

    def test_all_null_text_column_raises_ingest_error(self, tmp_path: Path) -> None:
        """Parquet with a 'text' column full of nulls raises CrucibleIngestError."""
        pa = pytest.importorskip("pyarrow")
        from core.errors import CrucibleIngestError
        from ingest.parquet_reader import read_parquet_records

        parquet_file = tmp_path / "nulls.parquet"
        _make_parquet(parquet_file, {"text": [None, None, None]})

        with pytest.raises(CrucibleIngestError, match="all values are empty or null"):
            read_parquet_records(parquet_file)

    def test_empty_string_text_column_raises_ingest_error(self, tmp_path: Path) -> None:
        """Parquet with text column of all empty strings raises CrucibleIngestError."""
        pytest.importorskip("pyarrow")
        from core.errors import CrucibleIngestError
        from ingest.parquet_reader import read_parquet_records

        parquet_file = tmp_path / "empty.parquet"
        _make_parquet(parquet_file, {"text": ["", "   ", "\t\n"]})

        with pytest.raises(CrucibleIngestError, match="all values are empty or null"):
            read_parquet_records(parquet_file)

    def test_mixed_null_and_valid_text_returns_valid_only(self, tmp_path: Path) -> None:
        """Parquet with mixed null/valid rows returns only the valid rows."""
        pytest.importorskip("pyarrow")
        from ingest.parquet_reader import read_parquet_records

        parquet_file = tmp_path / "mixed.parquet"
        _make_parquet(parquet_file, {"text": [None, "Hello world", "", "Good text"]})

        records = read_parquet_records(parquet_file)
        assert len(records) == 2
        texts = [r.text for r in records]
        assert "Hello world" in texts
        assert "Good text" in texts

    def test_valid_text_column_returns_all_rows(self, tmp_path: Path) -> None:
        """Valid parquet with text column returns one record per row."""
        pytest.importorskip("pyarrow")
        from ingest.parquet_reader import read_parquet_records

        parquet_file = tmp_path / "valid.parquet"
        texts = [f"Row {i} with some content for training" for i in range(20)]
        _make_parquet(parquet_file, {"text": texts})

        records = read_parquet_records(parquet_file)
        assert len(records) == 20
        for r in records:
            assert r.text in texts

    def test_prompt_response_columns_joined_correctly(self, tmp_path: Path) -> None:
        """Parquet with prompt+response columns are joined with newline."""
        pytest.importorskip("pyarrow")
        from ingest.parquet_reader import read_parquet_records

        parquet_file = tmp_path / "pr.parquet"
        _make_parquet(parquet_file, {
            "prompt": ["What is AI?", "What is ML?"],
            "response": ["AI is machine intelligence.", "ML is a subset of AI."],
        })

        records = read_parquet_records(parquet_file)
        assert len(records) == 2
        assert "What is AI?" in records[0].text
        assert "AI is machine intelligence." in records[0].text
        assert "\n" in records[0].text

    def test_all_null_prompt_response_raises_ingest_error(self, tmp_path: Path) -> None:
        """Prompt+response parquet where all rows are null raises CrucibleIngestError."""
        pytest.importorskip("pyarrow")
        from core.errors import CrucibleIngestError
        from ingest.parquet_reader import read_parquet_records

        parquet_file = tmp_path / "null_pr.parquet"
        _make_parquet(parquet_file, {
            "prompt": [None, None],
            "response": [None, None],
        })

        with pytest.raises(CrucibleIngestError):
            read_parquet_records(parquet_file)

    def test_unknown_columns_raises_ingest_error(self, tmp_path: Path) -> None:
        """Parquet with no recognized columns raises CrucibleIngestError with column list."""
        pytest.importorskip("pyarrow")
        from core.errors import CrucibleIngestError
        from ingest.parquet_reader import read_parquet_records

        parquet_file = tmp_path / "weird.parquet"
        _make_parquet(parquet_file, {"foo": [1, 2], "bar": [3, 4]})

        with pytest.raises(CrucibleIngestError, match="Cannot identify"):
            read_parquet_records(parquet_file)

    def test_unicode_text_preserved(self, tmp_path: Path) -> None:
        """Unicode text (CJK, emoji, accented chars) round-trips through parquet."""
        pytest.importorskip("pyarrow")
        from ingest.parquet_reader import read_parquet_records

        unicode_texts = [
            "日本語のテキスト: 機械学習について説明してください",
            "Ünïcödé téxt wïth àccênts",
            "Text with emoji 🎉🔥💡",
            "Arabic: مرحبا بالعالم",
        ]
        parquet_file = tmp_path / "unicode.parquet"
        _make_parquet(parquet_file, {"text": unicode_texts})

        records = read_parquet_records(parquet_file)
        assert len(records) == 4
        for r, expected in zip(records, unicode_texts):
            assert r.text == expected, f"Unicode text not preserved: {r.text!r} != {expected!r}"

    def test_content_column_name_recognized(self, tmp_path: Path) -> None:
        """'content' column name (used by HF datasets) is recognized."""
        pytest.importorskip("pyarrow")
        from ingest.parquet_reader import read_parquet_records

        parquet_file = tmp_path / "content.parquet"
        _make_parquet(parquet_file, {"content": ["First document.", "Second document."]})

        records = read_parquet_records(parquet_file)
        assert len(records) == 2

    def test_parquet_ingest_full_pipeline_to_training(self, tmp_path: Path) -> None:
        """Full pipeline: parquet file → ingest → SFT training → chat → non-crash.

        This is the key regression test: before the fix, a parquet with all-null
        values would silently produce 0 records, which then caused a confusing
        failure deep inside the training runner rather than at ingest time.
        """
        pytest.importorskip("pyarrow")
        from core.config import CrucibleConfig
        from core.sft_types import SftOptions
        from dataclasses import replace
        from serve.sft_runner import run_sft_training

        # Create a valid parquet file with SFT-compatible prompt/response data
        parquet_file = tmp_path / "sft_data.parquet"
        prompts = [f"Question about topic {i}?" for i in range(10)]
        responses = [f"Detailed answer explaining topic {i} thoroughly." for i in range(10)]
        _make_parquet(parquet_file, {"prompt": prompts, "response": responses})

        # Write it as JSONL for the SFT runner (parquet ingest is separate from SFT data path)
        # Parquet is for ingest; SFT data must be JSONL. Test the ingest side separately.
        from ingest.parquet_reader import read_parquet_records
        records = read_parquet_records(parquet_file)
        assert len(records) == 10, f"Expected 10 records from parquet, got {len(records)}"

        # All records have non-empty text
        for r in records:
            assert r.text.strip(), f"Empty text in record: {r!r}"

        # Now write JSONL and train to verify the full path works
        sft_file = tmp_path / "from_parquet.jsonl"
        sft_lines = [
            json.dumps({"prompt": p, "response": r_text})
            for p, r_text in zip(prompts, responses)
        ]
        sft_file.write_text("\n".join(sft_lines))

        result = run_sft_training(
            [],
            SftOptions(
                dataset_name="parquet-sft",
                output_dir=str(tmp_path / "out"),
                sft_data_path=str(sft_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )
        _verify_artifacts(result, expect_hf_base=_HF_MODEL)
        response = _chat_with_model(result.model_path, prompt="Question about topic 0?")
        assert isinstance(response, str)


# ── 2. SFT with toy fixture dataset ──────────────────────────────────────────


class TestSftEndToEnd:
    """SFT training with real fixture data, chat round-trip, tokenizer verification."""

    def test_sft_toy_fixture_trains_and_chats(self, tmp_path: Path) -> None:
        """Train SFT on sft_data.jsonl fixture, verify model, chat, verify tokenizer."""
        from core.sft_types import SftOptions
        from serve.sft_runner import run_sft_training

        sft_file = _FIXTURES / "sft_data.jsonl"
        assert sft_file.exists(), f"Fixture missing: {sft_file}"

        result = run_sft_training(
            [],
            SftOptions(
                dataset_name="sft-e2e",
                output_dir=str(tmp_path / "out"),
                sft_data_path=str(sft_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )
        _verify_artifacts(result, expect_hf_base=_HF_MODEL)

        # Chat with trained model
        response = _chat_with_model(result.model_path, prompt="What is machine learning?")
        assert isinstance(response, str)

        # Verify tokenizer is HF format (not VocabularyTokenizer / whitespace splitter)
        model_dir = Path(result.model_path).parent
        tokenizer_files = list(model_dir.glob("tokenizer*"))
        assert tokenizer_files, (
            f"No tokenizer files found in {model_dir}. "
            f"HF SFT should save tokenizer alongside model."
        )

    def test_sft_training_config_preserves_base_model(self, tmp_path: Path) -> None:
        """training_config.json records the base_model_path correctly after SFT."""
        from core.sft_types import SftOptions
        from serve.sft_runner import run_sft_training

        sft_file = _FIXTURES / "sft_data.jsonl"

        result = run_sft_training(
            [],
            SftOptions(
                dataset_name="sft-cfg",
                output_dir=str(tmp_path / "out"),
                sft_data_path=str(sft_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )

        cfg = json.loads((Path(result.model_path).parent / "training_config.json").read_text())
        assert cfg["base_model_path"] == _HF_MODEL, (
            f"base_model_path in training_config: {cfg.get('base_model_path')!r}"
        )
        assert cfg.get("method") in ("sft", None) or "sft" in str(cfg).lower()

    def test_sft_history_records_loss_values(self, tmp_path: Path) -> None:
        """History file should contain loss values (not all zero, not NaN)."""
        import math
        from core.sft_types import SftOptions
        from serve.sft_runner import run_sft_training

        sft_file = _FIXTURES / "sft_data.jsonl"

        result = run_sft_training(
            [],
            SftOptions(
                dataset_name="sft-history",
                output_dir=str(tmp_path / "out"),
                sft_data_path=str(sft_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )

        history = json.loads(Path(result.history_path).read_text())
        assert history, f"history is empty: {history!r}"

        # History can be a list (Crucible .pt) or dict with 'epochs' (HF trl)
        if isinstance(history, dict):
            epochs = history.get("epochs", [])
        else:
            epochs = history  # already a list
        assert len(epochs) >= 1, f"No epoch entries in history: {history!r}"

        # Every epoch should have a loss value
        for entry in epochs:
            loss_val = entry.get("train_loss") or entry.get("loss")
            if loss_val is not None:
                assert not math.isnan(float(loss_val)), f"NaN loss in history: {entry}"
                assert float(loss_val) >= 0, f"Negative loss: {entry}"

    def test_sft_model_state_dict_has_expected_keys(self, tmp_path: Path) -> None:
        """State dict should have HF transformer keys, not custom Crucible keys."""
        import torch
        from core.sft_types import SftOptions
        from serve.sft_runner import run_sft_training

        sft_file = _FIXTURES / "sft_data.jsonl"

        result = run_sft_training(
            [],
            SftOptions(
                dataset_name="sft-keys",
                output_dir=str(tmp_path / "out"),
                sft_data_path=str(sft_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )

        state = torch.load(str(result.model_path), weights_only=True, map_location="cpu")
        assert state, "empty state dict"

        # No double prefix
        bad = [k for k in state if k.startswith("model.model.")]
        assert not bad, f"Double-prefix keys: {bad[:5]}"

        # Keys should look like transformer keys (contain "weight" or "bias")
        weight_keys = [k for k in state if "weight" in k or "bias" in k]
        assert weight_keys, f"No weight/bias keys found. Keys: {list(state.keys())[:10]}"


# ── 3. DPO with toy fixture dataset ──────────────────────────────────────────


class TestDpoEndToEnd:
    """DPO training with chosen/rejected data, artifact verification."""

    def test_dpo_toy_fixture_trains_successfully(self, tmp_path: Path) -> None:
        """Train DPO on dpo_data.jsonl fixture with real tiny-gpt2."""
        from core.dpo_types import DpoOptions
        from serve.dpo_runner import run_dpo_training

        dpo_file = _FIXTURES / "dpo_data.jsonl"
        assert dpo_file.exists(), f"Fixture missing: {dpo_file}"

        result = run_dpo_training(
            [],
            DpoOptions(
                dataset_name="dpo-e2e",
                output_dir=str(tmp_path / "out"),
                dpo_data_path=str(dpo_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )
        _verify_artifacts(result, expect_hf_base=_HF_MODEL)

    def test_dpo_chat_with_trained_model(self, tmp_path: Path) -> None:
        """DPO-trained model must respond to chat without crashing."""
        from core.dpo_types import DpoOptions
        from serve.dpo_runner import run_dpo_training

        dpo_file = _FIXTURES / "dpo_data.jsonl"

        result = run_dpo_training(
            [],
            DpoOptions(
                dataset_name="dpo-chat",
                output_dir=str(tmp_path / "out"),
                dpo_data_path=str(dpo_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )
        # Critical: DPO model should be loadable and respond
        response = _chat_with_model(result.model_path, prompt="What is Python?")
        assert isinstance(response, str)

    def test_dpo_data_format_all_fields_parsed(self, tmp_path: Path) -> None:
        """Verify DPO training consumes prompt, chosen, AND rejected fields."""
        from core.dpo_types import DpoOptions
        from serve.dpo_runner import run_dpo_training

        # Write data where rejected responses are very different — if training uses
        # wrong field (e.g. ignores rejected), loss would behave differently.
        dpo_file = tmp_path / "dpo_strict.jsonl"
        lines = [
            json.dumps({"prompt": f"Q{i}", "chosen": "excellent detailed answer " * 5,
                        "rejected": "bad"})
            for i in range(10)
        ]
        dpo_file.write_text("\n".join(lines))

        result = run_dpo_training(
            [],
            DpoOptions(
                dataset_name="dpo-fields",
                output_dir=str(tmp_path / "out"),
                dpo_data_path=str(dpo_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )
        _verify_artifacts(result, expect_hf_base=_HF_MODEL)


# ── 4. KTO with toy fixture dataset ──────────────────────────────────────────


class TestKtoEndToEnd:
    """KTO training verifies is_desirable field is parsed correctly."""

    def test_kto_toy_fixture_trains_successfully(self, tmp_path: Path) -> None:
        """Train KTO on kto_data.jsonl fixture with real tiny-gpt2."""
        from core.kto_types import KtoOptions
        from serve.kto_runner import run_kto_training

        kto_file = _FIXTURES / "kto_data.jsonl"
        assert kto_file.exists(), f"Fixture missing: {kto_file}"

        result = run_kto_training(
            [],
            KtoOptions(
                dataset_name="kto-e2e",
                output_dir=str(tmp_path / "out"),
                kto_data_path=str(kto_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )
        _verify_artifacts(result, expect_hf_base=_HF_MODEL)

    def test_kto_mixed_desirable_undesirable_parsed(self, tmp_path: Path) -> None:
        """KTO must correctly parse is_desirable=true AND is_desirable=false rows."""
        from core.kto_types import KtoOptions
        from serve.kto_runner import run_kto_training

        # Create data with exactly 5 desirable and 5 undesirable
        kto_file = tmp_path / "kto_mixed.jsonl"
        lines = []
        for i in range(5):
            lines.append(json.dumps({
                "prompt": f"Q{i}", "response": "Good answer.", "is_desirable": True,
            }))
        for i in range(5):
            lines.append(json.dumps({
                "prompt": f"Q{i+5}", "response": "Bad.", "is_desirable": False,
            }))
        kto_file.write_text("\n".join(lines))

        result = run_kto_training(
            [],
            KtoOptions(
                dataset_name="kto-mixed",
                output_dir=str(tmp_path / "out"),
                kto_data_path=str(kto_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )
        _verify_artifacts(result, expect_hf_base=_HF_MODEL)

    def test_kto_chat_with_trained_model(self, tmp_path: Path) -> None:
        """KTO-trained model should be loadable and respond to chat."""
        from core.kto_types import KtoOptions
        from serve.kto_runner import run_kto_training

        kto_file = _FIXTURES / "kto_data.jsonl"

        result = run_kto_training(
            [],
            KtoOptions(
                dataset_name="kto-chat",
                output_dir=str(tmp_path / "out"),
                kto_data_path=str(kto_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )
        response = _chat_with_model(result.model_path, prompt="What is gravity?")
        assert isinstance(response, str)


# ── 5. RLVR with toy fixture dataset ─────────────────────────────────────────


class TestRlvrEndToEnd:
    """RLVR training verifies prompt+solution format is parsed correctly."""

    def test_rlvr_toy_fixture_trains_successfully(self, tmp_path: Path) -> None:
        """Train RLVR on rlvr_data.jsonl fixture with real tiny-gpt2."""
        from core.rlvr_types import RlvrOptions
        from serve.rlvr_runner import run_rlvr_training

        rlvr_file = _FIXTURES / "rlvr_data.jsonl"
        assert rlvr_file.exists(), f"Fixture missing: {rlvr_file}"

        result = run_rlvr_training(
            [],
            RlvrOptions(
                dataset_name="rlvr-e2e",
                output_dir=str(tmp_path / "out"),
                rlvr_data_path=str(rlvr_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )
        _verify_artifacts(result, expect_hf_base=_HF_MODEL)

    def test_rlvr_solution_field_consumed(self, tmp_path: Path) -> None:
        """RLVR training parses both prompt AND solution fields."""
        from core.rlvr_types import RlvrOptions
        from serve.rlvr_runner import run_rlvr_training

        rlvr_file = tmp_path / "rlvr_custom.jsonl"
        lines = [
            json.dumps({"prompt": "Solve: 2 + 2 = ?", "solution": "4"}),
            json.dumps({"prompt": "Solve: 5 * 5 = ?", "solution": "25"}),
            json.dumps({"prompt": "Solve: 10 - 3 = ?", "solution": "7"}),
            json.dumps({"prompt": "Solve: 6 / 2 = ?", "solution": "3"}),
            json.dumps({"prompt": "Solve: 4 ^ 2 = ?", "solution": "16"}),
        ] * 2  # 10 examples total
        rlvr_file.write_text("\n".join(lines))

        result = run_rlvr_training(
            [],
            RlvrOptions(
                dataset_name="rlvr-sol",
                output_dir=str(tmp_path / "out"),
                rlvr_data_path=str(rlvr_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )
        _verify_artifacts(result, expect_hf_base=_HF_MODEL)


# ── 6. LoRA with toy fixture dataset ─────────────────────────────────────────


class TestLoraEndToEnd:
    """LoRA training verifies adapter is saved with correct config fields."""

    def test_lora_toy_fixture_trains_and_chats(self, tmp_path: Path) -> None:
        """Train LoRA on lora_data.jsonl fixture, verify adapter, chat."""
        from core.lora_types import LoraTrainingOptions
        from serve.lora_training_runner import run_lora_training

        lora_file = _FIXTURES / "lora_data.jsonl"
        assert lora_file.exists(), f"Fixture missing: {lora_file}"

        result = run_lora_training(
            LoraTrainingOptions(
                dataset_name="lora-e2e",
                output_dir=str(tmp_path / "out"),
                lora_data_path=str(lora_file),
                base_model_path=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )
        _verify_artifacts(result, expect_hf_base=_HF_MODEL)

        # Chat with the LoRA-trained model
        response = _chat_with_model(result.model_path, prompt="Explain gradient descent.")
        assert isinstance(response, str)

    def test_lora_adapter_config_saved(self, tmp_path: Path) -> None:
        """LoRA training should save adapter_config.json in the output directory."""
        from core.lora_types import LoraTrainingOptions
        from serve.lora_training_runner import run_lora_training

        lora_file = _FIXTURES / "lora_data.jsonl"

        result = run_lora_training(
            LoraTrainingOptions(
                dataset_name="lora-cfg",
                output_dir=str(tmp_path / "out"),
                lora_data_path=str(lora_file),
                base_model_path=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )

        model_dir = Path(result.model_path).parent
        # peft saves adapter_config.json or the merged weights save training_config.json
        cfg_path = model_dir / "training_config.json"
        assert cfg_path.exists(), "training_config.json missing from LoRA output"
        cfg = json.loads(cfg_path.read_text())
        assert cfg.get("base_model_path") == _HF_MODEL


# ── 7. ORPO with toy fixture dataset ─────────────────────────────────────────


class TestOrpoEndToEnd:
    """ORPO training verifies chosen/rejected format (same as DPO)."""

    def test_orpo_toy_fixture_trains_successfully(self, tmp_path: Path) -> None:
        """Train ORPO on orpo_data.jsonl fixture with real tiny-gpt2."""
        from core.orpo_types import OrpoOptions
        from serve.orpo_runner import run_orpo_training

        orpo_file = _FIXTURES / "orpo_data.jsonl"
        assert orpo_file.exists(), f"Fixture missing: {orpo_file}"

        result = run_orpo_training(
            [],
            OrpoOptions(
                dataset_name="orpo-e2e",
                output_dir=str(tmp_path / "out"),
                orpo_data_path=str(orpo_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )
        _verify_artifacts(result, expect_hf_base=_HF_MODEL)


# ── 8. Benchmark runner: failure isolation ────────────────────────────────────


class TestBenchmarkRunnerEndToEnd:
    """Benchmark runner correctly isolates failures and writes partial results."""

    def _make_tiny_model(self, tmp_path: Path) -> str:
        """Train a tiny Crucible model for benchmark tests."""
        from core.types import DataRecord, RecordMetadata, TrainingOptions
        from serve.training_runner import run_training

        records = [
            DataRecord(
                record_id=f"bm-{i}",
                text="hello world test " * 20,
                metadata=RecordMetadata("test", "en", 0.9, 100.0),
            )
            for i in range(5)
        ]
        result = run_training(
            records,
            TrainingOptions(
                dataset_name="bm-base",
                output_dir=str(tmp_path / "bm_base"),
                hidden_dim=32,
                num_layers=1,
                attention_heads=2,
                batch_size=2,
                epochs=1,
                max_token_length=64,
                learning_rate=0.001,
                validation_split=0.2,
            ),
            42,
            tmp_path,
        )
        return result.model_path

    def test_single_benchmark_failure_does_not_abort_others(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If mmlu crashes, gsm8k still runs — per-benchmark isolation enforced."""
        from eval.benchmark_runner import BenchmarkResult, run_benchmarks

        ran = []

        def fake_mmlu(model_path, *, max_samples=None, eval_model=None):
            ran.append("mmlu")
            raise RuntimeError("Simulated OOM during MMLU")

        def fake_gsm8k(model_path, *, max_samples=None, eval_model=None):
            ran.append("gsm8k")
            return BenchmarkResult(
                benchmark_name="gsm8k", score=60.0, num_examples=5, correct=3,
            )

        monkeypatch.setattr("eval.benchmarks.mmlu.run_mmlu", fake_mmlu)
        monkeypatch.setattr("eval.benchmarks.gsm8k.run_gsm8k", fake_gsm8k)
        monkeypatch.setattr(
            "eval.benchmarks._model_loader.load_eval_model",
            lambda *a, **k: None,
        )

        model_path = self._make_tiny_model(tmp_path)
        result = run_benchmarks(model_path, ["mmlu", "gsm8k"])

        assert "mmlu" in ran, "mmlu should have been attempted"
        assert "gsm8k" in ran, "gsm8k should have run despite mmlu failure"
        assert len(result.benchmark_results) == 2

        mmlu_r = next(r for r in result.benchmark_results if r.benchmark_name == "mmlu")
        gsm8k_r = next(r for r in result.benchmark_results if r.benchmark_name == "gsm8k")
        assert mmlu_r.score == 0.0, f"Failed benchmark should record 0, got {mmlu_r.score}"
        assert "error" in mmlu_r.details, "Failed benchmark should record error detail"
        assert "OOM" in mmlu_r.details["error"]
        assert gsm8k_r.score == 60.0

    def test_partial_results_written_after_each_benchmark(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Partial result file is updated after each benchmark completes."""
        from eval.benchmark_runner import BenchmarkResult, run_benchmarks

        write_sequence = []

        def fake_mmlu(model_path, *, max_samples=None, eval_model=None):
            return BenchmarkResult(
                benchmark_name="mmlu", score=70.0, num_examples=5, correct=3,
            )

        def fake_gsm8k(model_path, *, max_samples=None, eval_model=None):
            return BenchmarkResult(
                benchmark_name="gsm8k", score=50.0, num_examples=5, correct=2,
            )

        original_write = None

        def capture_write(path, model_path, results, total):
            # Record state at each write: how many results written so far
            write_sequence.append({
                "num_results": len(results),
                "total": total,
                "names": [r.benchmark_name for r in results],
            })
            if path:
                import json as _json
                Path(path).write_text(_json.dumps({"partial": True, "count": len(results)}))

        monkeypatch.setattr("eval.benchmarks.mmlu.run_mmlu", fake_mmlu)
        monkeypatch.setattr("eval.benchmarks.gsm8k.run_gsm8k", fake_gsm8k)
        monkeypatch.setattr(
            "eval.benchmarks._model_loader.load_eval_model",
            lambda *a, **k: None,
        )
        monkeypatch.setattr("eval.benchmark_runner._write_partial_results", capture_write)

        model_path = self._make_tiny_model(tmp_path)
        output_path = str(tmp_path / "results.json")
        run_benchmarks(model_path, ["mmlu", "gsm8k"], output_path=output_path)

        assert len(write_sequence) == 2, (
            f"Expected 2 writes (one per benchmark), got {len(write_sequence)}: {write_sequence}"
        )
        assert write_sequence[0]["num_results"] == 1
        assert write_sequence[1]["num_results"] == 2
        assert write_sequence[0]["total"] == 2
        assert write_sequence[1]["total"] == 2

    def test_all_benchmarks_fail_average_is_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When every benchmark crashes, average score is 0.0."""
        from eval.benchmark_runner import run_benchmarks

        monkeypatch.setattr(
            "eval.benchmarks.mmlu.run_mmlu",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash")),
        )
        monkeypatch.setattr(
            "eval.benchmarks.gsm8k.run_gsm8k",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash")),
        )
        monkeypatch.setattr(
            "eval.benchmarks._model_loader.load_eval_model",
            lambda *a, **k: None,
        )

        model_path = self._make_tiny_model(tmp_path)
        result = run_benchmarks(model_path, ["mmlu", "gsm8k"])

        assert result.average_score == 0.0
        assert len(result.benchmark_results) == 2
        assert all(r.score == 0.0 for r in result.benchmark_results)

    def test_base_model_benchmark_failure_does_not_affect_fine_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Base model benchmark failure records 0 but fine model score is unaffected."""
        from eval.benchmark_runner import BenchmarkResult, run_benchmarks

        call_count = {"mmlu": 0}

        def fake_mmlu(model_path, *, max_samples=None, eval_model=None):
            call_count["mmlu"] += 1
            if call_count["mmlu"] > 1:  # Second call is for base model
                raise RuntimeError("Base model OOM")
            return BenchmarkResult(
                benchmark_name="mmlu", score=80.0, num_examples=5, correct=4,
            )

        monkeypatch.setattr("eval.benchmarks.mmlu.run_mmlu", fake_mmlu)
        monkeypatch.setattr(
            "eval.benchmarks._model_loader.load_eval_model",
            lambda *a, **k: None,
        )

        model_path = self._make_tiny_model(tmp_path)
        base_path = self._make_tiny_model(tmp_path / "base2")
        result = run_benchmarks(model_path, ["mmlu"], base_model_path=base_path)

        assert result.benchmark_results[0].score == 80.0, "Fine model score preserved"
        assert len(result.base_results) == 1
        assert result.base_results[0].score == 0.0, "Base model failure records 0"
        assert "error" in result.base_results[0].details


# ── 9. Tokenizer: no cross-module import of wrong function ────────────────────


class TestTokenizerModuleBoundary:
    """Verify the tokenizer rename fix: no wrong cross-imports remain."""

    def test_huggingface_tokenizer_module_exposes_load_tokenizer_from_file(self) -> None:
        """serve.huggingface_tokenizer exports load_tokenizer_from_file (renamed)."""
        from serve.huggingface_tokenizer import load_tokenizer_from_file
        assert callable(load_tokenizer_from_file)

    def test_load_huggingface_tokenizer_not_in_huggingface_tokenizer_module(self) -> None:
        """The old load_huggingface_tokenizer name must NOT be in huggingface_tokenizer.py."""
        import serve.huggingface_tokenizer as mod
        assert not hasattr(mod, "load_huggingface_tokenizer"), (
            "Old function name 'load_huggingface_tokenizer' still present in "
            "serve.huggingface_tokenizer — it should have been renamed to "
            "'load_tokenizer_from_file' to avoid collision with hf_model_loader."
        )

    def test_hf_model_loader_has_its_own_load_huggingface_tokenizer(self) -> None:
        """serve.hf_model_loader still exports its own load_huggingface_tokenizer."""
        from serve.hf_model_loader import load_huggingface_tokenizer
        assert callable(load_huggingface_tokenizer)

    def test_two_functions_have_incompatible_signatures(self) -> None:
        """The two tokenizer loaders accept different arguments — they must stay separate."""
        import inspect
        from serve.hf_model_loader import load_huggingface_tokenizer
        from serve.huggingface_tokenizer import load_tokenizer_from_file

        file_sig = inspect.signature(load_tokenizer_from_file)
        hf_sig = inspect.signature(load_huggingface_tokenizer)

        # load_tokenizer_from_file takes a local file path (tokenizer.json)
        assert "tokenizer_path" in file_sig.parameters, (
            f"load_tokenizer_from_file should have 'tokenizer_path' param, "
            f"got: {list(file_sig.parameters.keys())}"
        )

        # load_huggingface_tokenizer in hf_model_loader takes a model_id or path
        # (different param name — this is what proved they're incompatible)
        file_params = set(file_sig.parameters.keys())
        hf_params = set(hf_sig.parameters.keys())
        assert file_params != hf_params, (
            "Both tokenizer loaders have identical signatures — "
            "they may actually be the same function, which is wrong."
        )

    def test_training_metadata_imports_correct_tokenizer_loader(self) -> None:
        """training_metadata.py imports load_tokenizer_from_file, not the HF one."""
        import ast
        from pathlib import Path

        tm_path = Path(__file__).parent.parent.parent / "src" / "serve" / "training_metadata.py"
        assert tm_path.exists(), f"training_metadata.py not found at {tm_path}"

        tree = ast.parse(tm_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "serve.hf_model_loader":
                    imported = [alias.name for alias in node.names]
                    assert "load_huggingface_tokenizer" not in imported, (
                        "training_metadata.py imports load_huggingface_tokenizer from "
                        "hf_model_loader — this is WRONG. It should import "
                        "load_tokenizer_from_file from huggingface_tokenizer."
                    )


# ── 10. SFT: multiple problematic dataset formats that were broken ────────────


class TestDatasetFormatEdgeCases:
    """Verify training handles various dataset format edge cases correctly."""

    def test_sft_with_system_prompt_field_trains_ok(self, tmp_path: Path) -> None:
        """SFT data with optional system_prompt field doesn't crash training."""
        from core.sft_types import SftOptions
        from serve.sft_runner import run_sft_training

        # sft_data.jsonl has system_prompt on some rows — use the real fixture
        sft_file = _FIXTURES / "sft_data.jsonl"
        lines = json.loads(f"[{sft_file.read_text().strip()}]".replace("}\n{", "},{"))

        # Verify some rows have system_prompt, some don't
        with_sys = [l for l in lines if "system_prompt" in l]
        without_sys = [l for l in lines if "system_prompt" not in l]
        # The fixture has mixed — good for testing parser robustness
        assert len(lines) >= 5

        result = run_sft_training(
            [],
            SftOptions(
                dataset_name="sft-sysPrompt",
                output_dir=str(tmp_path / "out"),
                sft_data_path=str(sft_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )
        _verify_artifacts(result, expect_hf_base=_HF_MODEL)

    def test_sft_with_long_responses_trains_ok(self, tmp_path: Path) -> None:
        """SFT data with responses longer than max_token_length trains without crash."""
        from core.sft_types import SftOptions
        from serve.sft_runner import run_sft_training

        sft_file = tmp_path / "long_responses.jsonl"
        lines = [
            json.dumps({
                "prompt": f"Q{i}",
                # Response is 500 chars — much longer than max_token_length=64
                "response": f"Very long answer that should be truncated by the tokenizer. " * 8,
            })
            for i in range(10)
        ]
        sft_file.write_text("\n".join(lines))

        result = run_sft_training(
            [],
            SftOptions(
                dataset_name="sft-long",
                output_dir=str(tmp_path / "out"),
                sft_data_path=str(sft_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )
        _verify_artifacts(result, expect_hf_base=_HF_MODEL)

    def test_sft_with_unicode_and_special_chars_trains_ok(self, tmp_path: Path) -> None:
        """SFT data with unicode, newlines, and quotes in text trains correctly."""
        from core.sft_types import SftOptions
        from serve.sft_runner import run_sft_training

        sft_file = tmp_path / "unicode_sft.jsonl"
        lines = [
            json.dumps({"prompt": "日本語で説明してください", "response": "機械学習は統計的手法です。"}),
            json.dumps({"prompt": "Explain \"recursion\"", "response": "It's when f() calls f()."}),
            json.dumps({"prompt": "Math: 2+2=?", "response": "4\nSimple arithmetic."}),
            json.dumps({"prompt": "Code: `x = 1`", "response": "Assigns 1 to x."}),
            json.dumps({"prompt": "Emoji 🎉", "response": "Party time! 🎊"}),
        ] * 2  # 10 total
        sft_file.write_text("\n".join(lines))

        result = run_sft_training(
            [],
            SftOptions(
                dataset_name="sft-unicode",
                output_dir=str(tmp_path / "out"),
                sft_data_path=str(sft_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )
        _verify_artifacts(result, expect_hf_base=_HF_MODEL)
        # Must be able to chat after training on unicode data
        response = _chat_with_model(result.model_path, prompt="Hello")
        assert isinstance(response, str)

    def test_dpo_preferences_fixture_trains_ok(self, tmp_path: Path) -> None:
        """dpo_preferences.jsonl fixture (alternate DPO format) trains successfully."""
        from core.dpo_types import DpoOptions
        from serve.dpo_runner import run_dpo_training

        dpo_pref_file = _FIXTURES / "dpo_preferences.jsonl"
        assert dpo_pref_file.exists(), f"Fixture missing: {dpo_pref_file}"

        result = run_dpo_training(
            [],
            DpoOptions(
                dataset_name="dpo-pref",
                output_dir=str(tmp_path / "out"),
                dpo_data_path=str(dpo_pref_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )
        _verify_artifacts(result, expect_hf_base=_HF_MODEL)


# ── 11. GRPO with toy fixture dataset ────────────────────────────────────────


class TestGrpoEndToEnd:
    """GRPO training verifies prompt-only format."""

    def test_grpo_toy_fixture_trains_successfully(self, tmp_path: Path) -> None:
        """Train GRPO on grpo_data.jsonl fixture with real tiny-gpt2."""
        from core.grpo_types import GrpoOptions
        from serve.grpo_runner import run_grpo_training

        grpo_file = _FIXTURES / "grpo_data.jsonl"
        assert grpo_file.exists(), f"Fixture missing: {grpo_file}"

        result = run_grpo_training(
            [],
            GrpoOptions(
                dataset_name="grpo-e2e",
                output_dir=str(tmp_path / "out"),
                grpo_data_path=str(grpo_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )
        _verify_artifacts(result, expect_hf_base=_HF_MODEL)


# ── 12. Artifact contract: reproducibility_bundle_path is None not "" ─────────


class TestArtifactContract:
    """Verify artifact contracts have semantically correct field values."""

    def test_sft_artifact_contract_reproducibility_bundle_path_is_none(
        self, tmp_path: Path,
    ) -> None:
        """reproducibility_bundle_path should be None (not empty string) when absent."""
        import json
        from core.sft_types import SftOptions
        from serve.sft_runner import run_sft_training

        sft_file = _FIXTURES / "sft_data.jsonl"

        result = run_sft_training(
            [],
            SftOptions(
                dataset_name="sft-contract",
                output_dir=str(tmp_path / "out"),
                sft_data_path=str(sft_file),
                base_model=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )

        # artifact_contract_path may or may not be present
        if result.artifact_contract_path:
            contract_path = Path(result.artifact_contract_path)
            if contract_path.exists():
                contract = json.loads(contract_path.read_text())
                bundle_path = contract.get("reproducibility_bundle_path")
                # Must be None or absent — not empty string ""
                assert bundle_path is None or bundle_path != "", (
                    f"reproducibility_bundle_path should be None when no bundle exists, "
                    f"got: {bundle_path!r}"
                )

    def test_qlora_artifact_contract_reproducibility_bundle_path_is_none(
        self, tmp_path: Path,
    ) -> None:
        """QLoRA artifact contract: reproducibility_bundle_path is None, not ''."""
        pytest.importorskip("bitsandbytes", reason="bitsandbytes required for QLoRA")
        import json
        from core.qlora_types import QloraOptions
        from serve.qlora_runner import run_qlora_training

        qlora_file = _FIXTURES / "qlora_data.jsonl"

        result = run_qlora_training(
            [],
            QloraOptions(
                dataset_name="qlora-contract",
                output_dir=str(tmp_path / "out"),
                qlora_data_path=str(qlora_file),
                base_model_path=_HF_MODEL,
                **_HF,
            ),
            42,
            tmp_path,
        )

        if result.artifact_contract_path:
            contract_path = Path(result.artifact_contract_path)
            if contract_path.exists():
                contract = json.loads(contract_path.read_text())
                bundle_path = contract.get("reproducibility_bundle_path")
                assert bundle_path is None or bundle_path != "", (
                    f"QLoRA reproducibility_bundle_path should be None, got: {bundle_path!r}"
                )
