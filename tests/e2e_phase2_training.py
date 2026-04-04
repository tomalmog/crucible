#!/usr/bin/env python3
"""Phase 2: End-to-end training tests for every Crucible training method.

Tests Crucible .pt training from scratch, all HF training methods with
sshleifer/tiny-gpt2, and edge cases (unicode, small dataset, long sequences).
"""

import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from core.types import DataRecord, RecordMetadata
from core.chat_types import ChatOptions
from store.dataset_sdk import CrucibleClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULTS: list[dict] = []
HF_MODEL = "sshleifer/tiny-gpt2"
HF_COMMON = dict(epochs=1, batch_size=2, max_token_length=64, validation_split=0.2, precision_mode="fp32")


def _meta(i: int) -> RecordMetadata:
    return RecordMetadata(
        source_uri=f"test://record-{i}",
        language="en",
        quality_score=1.0,
        perplexity=10.0,
    )


def make_records(texts: list[str]) -> list[DataRecord]:
    return [
        DataRecord(record_id=f"rec-{i}", text=t, metadata=_meta(i))
        for i, t in enumerate(texts)
    ]


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def verify_artifacts(output_dir: str, test_name: str, is_hf: bool = False) -> list[str]:
    """Verify common training artifacts. Returns list of issues."""
    issues = []
    od = Path(output_dir)

    # model.pt
    model_pt = od / "model.pt"
    if not model_pt.exists():
        issues.append("model.pt missing")
    elif model_pt.stat().st_size == 0:
        issues.append("model.pt is empty")
    else:
        sd = torch.load(model_pt, map_location="cpu", weights_only=True)
        if not sd:
            issues.append("model.pt state dict is empty")
        # Double-prefix check
        bad_keys = [k for k in sd if k.startswith("model.model.")]
        if bad_keys:
            issues.append(f"Double-prefix keys found: {bad_keys[:3]}")

    # training_config.json
    tc_path = od / "training_config.json"
    if not tc_path.exists():
        issues.append("training_config.json missing")
    else:
        tc = json.loads(tc_path.read_text())
        if is_hf and tc.get("base_model_path") != HF_MODEL:
            issues.append(f"training_config.json base_model_path={tc.get('base_model_path')!r}, expected {HF_MODEL!r}")

    # history.json
    hist_path = od / "history.json"
    if not hist_path.exists():
        issues.append("history.json missing")
    else:
        hist = json.loads(hist_path.read_text())
        if not hist:
            issues.append("history.json is empty")

    return issues


def try_chat(model_path: str, is_hf: bool = False) -> str | None:
    """Try chatting with a model. Returns error message or None on success."""
    try:
        client = CrucibleClient()
        opts = ChatOptions(
            model_path=model_path,
            prompt="Hello",
            max_new_tokens=5,
            max_token_length=64,
        )
        result = client.chat(opts)
        if not isinstance(result.response_text, str):
            return f"Chat response is not a string: {type(result.response_text)}"
        return None
    except Exception as e:
        return f"Chat failed: {e}"


def run_test(test_id: str, test_name: str, fn):
    """Run a single test, catch exceptions, record results."""
    print(f"\n{'='*70}")
    print(f"TEST {test_id}: {test_name}")
    print(f"{'='*70}")
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        RESULTS.append({"id": test_id, "name": test_name, "status": "PASS", "elapsed": f"{elapsed:.1f}s", "error": ""})
        print(f"  -> PASS ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        RESULTS.append({"id": test_id, "name": test_name, "status": "FAIL", "elapsed": f"{elapsed:.1f}s", "error": tb})
        print(f"  -> FAIL ({elapsed:.1f}s)")
        print(tb)


# ---------------------------------------------------------------------------
# 2.1 Crucible .pt Training
# ---------------------------------------------------------------------------

def test_2_1_1_basic_train():
    """Train a tiny Crucible model from scratch using text data."""
    from core.training_types import TrainingOptions

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models learn patterns from data.",
        "Neural networks consist of layers of interconnected nodes.",
        "Training a model requires a dataset and a loss function.",
        "The gradient descent algorithm updates model weights iteratively.",
    ]
    records = make_records(texts)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "basic_train_output")
        opts = TrainingOptions(
            dataset_name="__test__",
            output_dir=output_dir,
            hidden_dim=32,
            num_layers=1,
            attention_heads=2,
            epochs=1,
            batch_size=2,
            max_token_length=64,
            precision_mode="fp32",
        )
        from serve.training_runner import run_training
        result = run_training(
            records=records,
            options=opts,
            random_seed=42,
            data_root=Path(tmpdir),
        )
        # Verify artifacts
        issues = verify_artifacts(output_dir, "2.1.1", is_hf=False)

        # Check history has loss values
        hist = json.loads((Path(output_dir) / "history.json").read_text())
        if isinstance(hist, dict):
            epoch_metrics = hist.get("epoch_metrics", hist.get("epochs", []))
        else:
            epoch_metrics = hist
        if not epoch_metrics:
            issues.append("No epoch metrics in history.json")

        # Chat with model
        chat_err = try_chat(output_dir, is_hf=False)
        if chat_err:
            issues.append(chat_err)

        if issues:
            raise AssertionError("Artifact verification failed:\n  " + "\n  ".join(issues))
        print(f"  model.pt size: {(Path(output_dir) / 'model.pt').stat().st_size} bytes")
        print(f"  history epochs: {len(epoch_metrics)}")


# ---------------------------------------------------------------------------
# 2.2 HuggingFace Training Methods
# ---------------------------------------------------------------------------

def _make_sft_data(n: int = 10) -> list[dict]:
    return [{"prompt": f"Question {i}: What is {i}+{i}?", "response": f"The answer is {i*2}."} for i in range(n)]


def _make_dpo_data(n: int = 10) -> list[dict]:
    return [
        {"prompt": f"Q{i}: Explain {i}.", "chosen": f"Good answer about {i}.", "rejected": f"Bad answer about {i}."}
        for i in range(n)
    ]


def _make_kto_data(n: int = 10) -> list[dict]:
    return [
        {"prompt": f"Q{i}: What is {i}?", "response": f"Answer: {i}.", "is_desirable": i % 2 == 0}
        for i in range(n)
    ]


def _make_grpo_data(n: int = 10) -> list[dict]:
    return [{"prompt": f"Solve: what is {i} times 2?"} for i in range(n)]


def _make_rlvr_data(n: int = 10) -> list[dict]:
    return [{"prompt": f"What is {i} + {i}?", "solution": str(i * 2)} for i in range(n)]


def _hf_test(method_name: str, data_rows: list[dict], options_class, data_path_field: str,
             extra_kwargs: dict | None = None, test_chat: bool = True):
    """Generic HF training method test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data.jsonl")
        write_jsonl(data_path, data_rows)
        output_dir = os.path.join(tmpdir, f"{method_name}_output")

        kwargs = dict(
            dataset_name="",
            output_dir=output_dir,
            base_model=HF_MODEL,
            **{data_path_field: data_path},
            **HF_COMMON,
        )
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        # Some options classes use base_model_path instead of base_model
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(options_class)}
        if "base_model" not in valid_fields and "base_model_path" in valid_fields:
            kwargs["base_model_path"] = kwargs.pop("base_model")

        # Filter to valid fields only
        kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        opts = options_class(**kwargs)
        client = CrucibleClient()
        dispatch_name = {
            "sft": "sft_train",
            "lora": "lora_train",
            "qlora": "qlora_train",
            "dpo": "dpo_train",
            "kto": "kto_train",
            "orpo": "orpo_train",
            "grpo": "grpo_train",
            "rlvr": "rlvr_train",
        }[method_name]
        result = getattr(client, dispatch_name)(opts)

        issues = verify_artifacts(output_dir, method_name, is_hf=True)
        if issues:
            raise AssertionError(f"{method_name} artifact verification failed:\n  " + "\n  ".join(issues))

        # Chat test - reported separately
        if test_chat:
            chat_err = try_chat(output_dir, is_hf=True)
            if chat_err:
                print(f"  [CHAT BUG] {chat_err}")
                raise AssertionError(f"{method_name} chat failed (training OK, artifacts OK):\n  {chat_err}")

        print(f"  model.pt size: {(Path(output_dir) / 'model.pt').stat().st_size} bytes")


def test_2_2_1_sft():
    from core.sft_types import SftOptions
    _hf_test("sft", _make_sft_data(10), SftOptions, "sft_data_path")


def test_2_2_2_lora():
    from core.lora_types import LoraTrainingOptions
    _hf_test("lora", _make_sft_data(10), LoraTrainingOptions, "lora_data_path")


def test_2_2_3_qlora():
    from core.qlora_types import QloraOptions
    _hf_test("qlora", _make_sft_data(10), QloraOptions, "qlora_data_path", test_chat=True)


def test_2_2_4_dpo():
    from core.dpo_types import DpoOptions
    _hf_test("dpo", _make_dpo_data(10), DpoOptions, "dpo_data_path")


def test_2_2_5_kto():
    from core.kto_types import KtoOptions
    _hf_test("kto", _make_kto_data(10), KtoOptions, "kto_data_path")


def test_2_2_6_orpo():
    from core.orpo_types import OrpoOptions
    _hf_test("orpo", _make_dpo_data(10), OrpoOptions, "orpo_data_path")


def test_2_2_7_grpo():
    from core.grpo_types import GrpoOptions
    _hf_test("grpo", _make_grpo_data(10), GrpoOptions, "grpo_data_path")


def test_2_2_8_rlvr():
    from core.rlvr_types import RlvrOptions
    _hf_test("rlvr", _make_rlvr_data(10), RlvrOptions, "rlvr_data_path")


# ---------------------------------------------------------------------------
# 2.3 Edge Cases
# ---------------------------------------------------------------------------

def test_2_3_1_unicode():
    """SFT with Japanese/emoji prompts."""
    from core.sft_types import SftOptions
    data = [
        {"prompt": "日本語のテスト質問です", "response": "これは日本語の回答です。"},
        {"prompt": "What does the party emoji mean?", "response": "It means celebration!"},
        {"prompt": "Wie geht es Ihnen?", "response": "Mir geht es gut, danke!"},
        {"prompt": "数学の問題: 1+1は？", "response": "答えは2です。"},
        {"prompt": "Hello in Russian", "response": "Privet!"},
        {"prompt": "Arabic greeting", "response": "Marhaba!"},
        {"prompt": "World in Japanese", "response": "Hello World!"},
        {"prompt": "Test 8: emoji test", "response": "Rocket, fire, hundred!"},
        {"prompt": "Mixed test: content 123", "response": "Mixed response test 456"},
        {"prompt": "Final test", "response": "All done!"},
    ]
    _hf_test("sft", data, SftOptions, "sft_data_path", test_chat=True)


def test_2_3_2_small_dataset():
    """SFT with only 3 examples. Uses validation_split > 0 to avoid ValueError."""
    from core.sft_types import SftOptions
    data = [
        {"prompt": "What is 1+1?", "response": "2"},
        {"prompt": "What is 2+2?", "response": "4"},
        {"prompt": "What is 3+3?", "response": "6"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data.jsonl")
        write_jsonl(data_path, data)
        output_dir = os.path.join(tmpdir, "small_sft_output")
        opts = SftOptions(
            dataset_name="",
            output_dir=output_dir,
            sft_data_path=data_path,
            base_model=HF_MODEL,
            epochs=1,
            batch_size=2,
            max_token_length=64,
            validation_split=0.33,  # 1/3 for eval to avoid the 0.0 ValueError
            precision_mode="fp32",
        )
        client = CrucibleClient()
        result = client.sft_train(opts)
        issues = verify_artifacts(output_dir, "2.3.2", is_hf=True)
        if issues:
            raise AssertionError("Small dataset verification failed:\n  " + "\n  ".join(issues))

        chat_err = try_chat(output_dir, is_hf=True)
        if chat_err:
            raise AssertionError(f"Small dataset chat failed:\n  {chat_err}")
        print(f"  model.pt size: {(Path(output_dir) / 'model.pt').stat().st_size} bytes")


def test_2_3_2b_zero_val_split():
    """SFT with 3 examples and validation_split=0.0 -- tests edge case.

    HuggingFace datasets raises ValueError if test_size=0.0, so the runner
    should handle this gracefully (skip split or use all data for training).
    """
    from core.sft_types import SftOptions
    data = [
        {"prompt": "What is 1+1?", "response": "2"},
        {"prompt": "What is 2+2?", "response": "4"},
        {"prompt": "What is 3+3?", "response": "6"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data.jsonl")
        write_jsonl(data_path, data)
        output_dir = os.path.join(tmpdir, "small_sft_zero_val")
        opts = SftOptions(
            dataset_name="",
            output_dir=output_dir,
            sft_data_path=data_path,
            base_model=HF_MODEL,
            epochs=1,
            batch_size=2,
            max_token_length=64,
            validation_split=0.0,
            precision_mode="fp32",
        )
        client = CrucibleClient()
        result = client.sft_train(opts)
        issues = verify_artifacts(output_dir, "2.3.2b", is_hf=True)
        if issues:
            raise AssertionError("Zero val split verification failed:\n  " + "\n  ".join(issues))
        print(f"  model.pt size: {(Path(output_dir) / 'model.pt').stat().st_size} bytes")


def test_2_3_3_long_sequences():
    """SFT with responses longer than max_token_length to verify truncation."""
    from core.sft_types import SftOptions
    long_response = "word " * 500  # ~500 tokens, max_token_length=64
    data = [
        {"prompt": f"Generate long text {i}", "response": long_response}
        for i in range(10)
    ]
    _hf_test("sft", data, SftOptions, "sft_data_path", test_chat=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PHASE 2: End-to-End Training Tests")
    print("=" * 70)

    tests = [
        ("2.1.1", "Basic Crucible .pt train (from scratch)", test_2_1_1_basic_train),
        ("2.2.1", "SFT (sshleifer/tiny-gpt2)", test_2_2_1_sft),
        ("2.2.2", "LoRA (sshleifer/tiny-gpt2)", test_2_2_2_lora),
        ("2.2.3", "QLoRA (sshleifer/tiny-gpt2)", test_2_2_3_qlora),
        ("2.2.4", "DPO (sshleifer/tiny-gpt2)", test_2_2_4_dpo),
        ("2.2.5", "KTO (sshleifer/tiny-gpt2)", test_2_2_5_kto),
        ("2.2.6", "ORPO (sshleifer/tiny-gpt2)", test_2_2_6_orpo),
        ("2.2.7", "GRPO (sshleifer/tiny-gpt2)", test_2_2_7_grpo),
        ("2.2.8", "RLVR (sshleifer/tiny-gpt2)", test_2_2_8_rlvr),
        ("2.3.1", "Unicode in training data (SFT)", test_2_3_1_unicode),
        ("2.3.2", "Very small dataset (3 rows, SFT)", test_2_3_2_small_dataset),
        ("2.3.2b", "Zero validation_split (3 rows, SFT)", test_2_3_2b_zero_val_split),
        ("2.3.3", "Long sequences (SFT)", test_2_3_3_long_sequences),
    ]

    for test_id, test_name, fn in tests:
        run_test(test_id, test_name, fn)

    # Summary
    print("\n\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")
    print(f"\nTotal: {len(RESULTS)}  |  PASS: {passed}  |  FAIL: {failed}\n")
    print(f"{'ID':<8} {'Status':<8} {'Time':<10} {'Name'}")
    print("-" * 70)
    for r in RESULTS:
        print(f"{r['id']:<8} {r['status']:<8} {r['elapsed']:<10} {r['name']}")

    # Categorize failures
    chat_failures = []
    training_failures = []
    for r in RESULTS:
        if r["status"] == "FAIL":
            if "Chat failed" in r["error"] or "chat failed" in r["error"]:
                chat_failures.append(r)
            else:
                training_failures.append(r)

    if chat_failures:
        print(f"\nChat-only failures ({len(chat_failures)}) - training + artifacts OK:")
        for r in chat_failures:
            print(f"  {r['id']}: {r['name']}")
    if training_failures:
        print(f"\nTraining/artifact failures ({len(training_failures)}):")
        for r in training_failures:
            print(f"  {r['id']}: {r['name']}")

    # Write detailed results
    results_path = "/Users/tomalmog/programming/Febuary 2026/forge/test_results_phase2.txt"
    with open(results_path, "w") as f:
        f.write("PHASE 2: End-to-End Training Test Results\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total: {len(RESULTS)}  |  PASS: {passed}  |  FAIL: {failed}\n\n")
        f.write(f"{'ID':<8} {'Status':<8} {'Time':<10} {'Name'}\n")
        f.write("-" * 70 + "\n")
        for r in RESULTS:
            f.write(f"{r['id']:<8} {r['status']:<8} {r['elapsed']:<10} {r['name']}\n")

        # Bug summary
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("BUGS FOUND\n")
        f.write("=" * 70 + "\n\n")

        if chat_failures:
            f.write("BUG 1: Chat with HF-trained models fails when given directory path\n")
            f.write("-" * 70 + "\n")
            f.write("File: src/serve/hf_model_loader.py, function _load_custom_weights()\n")
            f.write("Symptom: 'Unsupported weight format '' for <dir>'\n")
            f.write("Root cause: _load_custom_weights() receives a directory path (the output\n")
            f.write("  directory) but only handles file paths. It needs to search inside the\n")
            f.write("  directory for model.pt or model.safetensors, similar to how\n")
            f.write("  _detect_dir_aware_format() in chat_runner.py resolves directories.\n")
            f.write("Impact: Chat with ANY HF-trained model (SFT, LoRA, QLoRA, DPO, KTO,\n")
            f.write("  ORPO, GRPO, RLVR) is broken when pointing at the output directory.\n")
            f.write("Affected tests: " + ", ".join(r['id'] for r in chat_failures) + "\n\n")

        if any("Zero val" in r["name"] or "validation_split" in r.get("error", "").lower() for r in RESULTS if r["status"] == "FAIL"):
            f.write("BUG 2: validation_split=0.0 crashes HF training runners\n")
            f.write("-" * 70 + "\n")
            f.write("File: src/serve/sft_runner.py (and likely other trl-based runners)\n")
            f.write("Symptom: ValueError from datasets.train_test_split(test_size=0.0)\n")
            f.write("Root cause: When validation_split=0.0, the runner should skip the\n")
            f.write("  train/test split and use all data for training. Currently it blindly\n")
            f.write("  passes 0.0 to train_test_split(), which is invalid.\n")
            f.write("Impact: Users who don't want a validation set get a crash.\n\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("FAILURE DETAILS\n")
        f.write("=" * 70 + "\n\n")
        for r in RESULTS:
            if r["status"] == "FAIL":
                f.write(f"--- {r['id']}: {r['name']} ---\n")
                f.write(r["error"] + "\n\n")

    print(f"\nResults written to: {results_path}")


if __name__ == "__main__":
    main()
