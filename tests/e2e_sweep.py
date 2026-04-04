#!/usr/bin/env python3
"""End-to-end hyperparameter sweep test suite for Crucible.

Tests full sweep execution across training methods, parameter strategies,
failure handling, method-args propagation, and model auto-registration.
Runs real training with tiny models — no mocks.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

_TMP_ROOT = tempfile.mkdtemp(prefix="crucible_sweep_e2e_")
os.environ["CRUCIBLE_DATA_ROOT"] = _TMP_ROOT

from core.config import CrucibleConfig
from core.sweep_types import SweepConfig, SweepParameter
from core.types import DataRecord, RecordMetadata
from store.dataset_sdk import CrucibleClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULTS: list[dict] = []
HF_MODEL = "sshleifer/tiny-gpt2"
HF_COMMON: dict[str, object] = dict(
    epochs=1, batch_size=2, max_token_length=64,
    validation_split=0.0, precision_mode="fp32",
)


def _meta(i: int) -> RecordMetadata:
    return RecordMetadata(
        source_uri=f"test://sweep-{i}",
        language="en",
        quality_score=1.0,
        perplexity=10.0,
    )


def make_records(texts: list[str]) -> list[DataRecord]:
    return [
        DataRecord(record_id=f"rec-{i}", text=t, metadata=_meta(i))
        for i, t in enumerate(texts)
    ]


def make_client() -> CrucibleClient:
    config = CrucibleConfig(
        data_root=Path(_TMP_ROOT),
        s3_region=None,
        s3_profile=None,
        random_seed=42,
    )
    return CrucibleClient(config)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ingest_dataset(client: CrucibleClient, name: str, rows: list[dict]) -> str:
    """Write rows to a temp JSONL file and ingest as a named dataset."""
    from core.types import IngestOptions

    tmpdir = tempfile.mkdtemp(prefix="sweep_ds_")
    path = Path(tmpdir) / "data.jsonl"
    write_jsonl(path, rows)
    opts = IngestOptions(dataset_name=name, source_uri=str(path))
    return client.ingest(opts)


def run_test(test_id: str, test_name: str, fn) -> None:
    """Run a single test, catch exceptions, record results."""
    print(f"\n{'=' * 70}")
    print(f"TEST {test_id}: {test_name}")
    print(f"{'=' * 70}")
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        RESULTS.append({
            "id": test_id, "name": test_name,
            "status": "PASS", "elapsed": f"{elapsed:.1f}s", "error": "",
        })
        print(f"  -> PASS ({elapsed:.1f}s)")
    except Exception:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        RESULTS.append({
            "id": test_id, "name": test_name,
            "status": "FAIL", "elapsed": f"{elapsed:.1f}s", "error": tb,
        })
        print(f"  -> FAIL ({elapsed:.1f}s)")
        print(tb)


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def _text_data(n: int = 10) -> list[dict]:
    return [{"text": f"Training document number {i} with enough text to be useful."} for i in range(n)]


def _sft_data(n: int = 10) -> list[dict]:
    return [{"prompt": f"Question {i}: What is {i}+{i}?", "response": f"The answer is {i * 2}."} for i in range(n)]


def _dpo_data(n: int = 10) -> list[dict]:
    return [
        {"prompt": f"Q{i}: Explain {i}.", "chosen": f"Good answer about {i}.", "rejected": f"Bad answer about {i}."}
        for i in range(n)
    ]


def _kto_data(n: int = 10) -> list[dict]:
    return [
        {"prompt": f"Q{i}: What is {i}?", "response": f"Answer: {i}.", "is_desirable": i % 2 == 0}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# S.1  Basic sweep: Crucible .pt model, grid search
# ---------------------------------------------------------------------------

def test_s_1_basic_sweep_grid():
    """Grid sweep over learning_rate with basic train method on .pt model."""
    from serve.sweep_runner import run_sweep

    client = make_client()
    ds_name = ingest_dataset(client, "sweep-basic-grid", _text_data(10))

    with tempfile.TemporaryDirectory() as tmpdir:
        config = SweepConfig(
            dataset_name=ds_name,
            output_dir=tmpdir,
            base_output_dir=tmpdir,
            parameters=(
                SweepParameter(name="learning_rate", values=(0.01, 0.001)),
            ),
            strategy="grid",
            max_trials=10,
            metric="train_loss",
            minimize=True,
            training_method="train",
            method_args=(
                ("hidden_dim", "32"),
                ("num_layers", "1"),
                ("attention_heads", "2"),
                ("epochs", "1"),
                ("batch_size", "2"),
                ("max_token_length", "64"),
                ("precision_mode", "fp32"),
            ),
        )
        result = run_sweep(client, config, random_seed=42)

        # Should have 2 trials (grid: 2 learning_rate values)
        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"
        # Best trial should be identified
        assert result.best_trial_id in (0, 1), f"Unexpected best_trial_id: {result.best_trial_id}"
        # Metric should be finite
        assert math.isfinite(result.best_metric_value), f"Non-finite metric: {result.best_metric_value}"
        # Model path should exist
        assert result.model_path, "No model_path in SweepResult"
        assert Path(result.model_path).exists(), f"Best model not found: {result.model_path}"
        # Each trial should have distinct parameters
        params_set = {t.parameters["learning_rate"] for t in result.trials}
        assert params_set == {0.01, 0.001}, f"Unexpected params: {params_set}"
        print(f"  best_trial={result.best_trial_id}, metric={result.best_metric_value:.6f}")


# ---------------------------------------------------------------------------
# S.2  Random search strategy
# ---------------------------------------------------------------------------

def test_s_2_random_strategy():
    """Random sweep with max_trials cap on .pt model."""
    from serve.sweep_runner import run_sweep

    client = make_client()
    ds_name = ingest_dataset(client, "sweep-random", _text_data(10))

    with tempfile.TemporaryDirectory() as tmpdir:
        config = SweepConfig(
            dataset_name=ds_name,
            output_dir=tmpdir,
            base_output_dir=tmpdir,
            parameters=(
                SweepParameter(
                    name="learning_rate", values=(),
                    min_value=0.0001, max_value=0.1, log_scale=True,
                ),
            ),
            strategy="random",
            max_trials=3,
            metric="train_loss",
            minimize=True,
            training_method="train",
            method_args=(
                ("hidden_dim", "32"),
                ("num_layers", "1"),
                ("attention_heads", "2"),
                ("epochs", "1"),
                ("batch_size", "2"),
                ("max_token_length", "64"),
                ("precision_mode", "fp32"),
            ),
        )
        result = run_sweep(client, config, random_seed=42)

        assert len(result.trials) == 3, f"Expected 3 trials, got {len(result.trials)}"
        # All learning rates should be unique and within bounds
        lrs = [t.parameters["learning_rate"] for t in result.trials]
        assert len(set(lrs)) == 3, f"Expected 3 unique learning rates, got {lrs}"
        for lr in lrs:
            assert 0.0001 <= lr <= 0.1, f"learning_rate {lr} out of bounds"
        assert math.isfinite(result.best_metric_value)
        print(f"  3 trials, lrs={[f'{lr:.6f}' for lr in lrs]}, best_metric={result.best_metric_value:.6f}")


# ---------------------------------------------------------------------------
# S.3  Multi-parameter grid (cartesian product)
# ---------------------------------------------------------------------------

def test_s_3_multi_param_grid():
    """Grid sweep over 2 parameters produces cartesian product of trials."""
    from serve.sweep_runner import run_sweep

    client = make_client()
    ds_name = ingest_dataset(client, "sweep-multi", _text_data(10))

    with tempfile.TemporaryDirectory() as tmpdir:
        config = SweepConfig(
            dataset_name=ds_name,
            output_dir=tmpdir,
            base_output_dir=tmpdir,
            parameters=(
                SweepParameter(name="learning_rate", values=(0.01, 0.001)),
                SweepParameter(name="batch_size", values=(2, 4)),
            ),
            strategy="grid",
            max_trials=10,
            metric="train_loss",
            minimize=True,
            training_method="train",
            method_args=(
                ("hidden_dim", "32"),
                ("num_layers", "1"),
                ("attention_heads", "2"),
                ("epochs", "1"),
                ("max_token_length", "64"),
                ("precision_mode", "fp32"),
            ),
        )
        result = run_sweep(client, config, random_seed=42)

        # 2 × 2 = 4 trials
        assert len(result.trials) == 4, f"Expected 4 trials, got {len(result.trials)}"
        # Verify all 4 parameter combos are present
        combos = {(t.parameters["learning_rate"], t.parameters["batch_size"]) for t in result.trials}
        expected = {(0.01, 2), (0.01, 4), (0.001, 2), (0.001, 4)}
        assert combos == expected, f"Missing combos: expected {expected}, got {combos}"
        print(f"  4 trials, best_metric={result.best_metric_value:.6f}")


# ---------------------------------------------------------------------------
# S.4  SFT sweep with trl (HuggingFace model)
# ---------------------------------------------------------------------------

def test_s_4_sft_sweep():
    """SFT sweep over learning_rate using trl SFTTrainer on tiny-gpt2."""
    from serve.sweep_runner import run_sweep

    client = make_client()
    ds_name = ingest_dataset(client, "sweep-sft", _sft_data(10))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write source data for SFT data-path resolution
        source_path = Path(tmpdir) / "sft_data.jsonl"
        write_jsonl(source_path, _sft_data(10))

        config = SweepConfig(
            dataset_name=ds_name,
            output_dir=tmpdir,
            base_output_dir=tmpdir,
            parameters=(
                SweepParameter(name="learning_rate", values=(5e-4, 1e-3)),
            ),
            strategy="grid",
            max_trials=10,
            metric="train_loss",
            minimize=True,
            training_method="sft",
            method_args=(
                ("base_model", HF_MODEL),
                ("sft_data_path", str(source_path)),
                ("epochs", "1"),
                ("batch_size", "2"),
                ("max_token_length", "64"),
                ("validation_split", "0.0"),
                ("precision_mode", "fp32"),
            ),
        )
        result = run_sweep(client, config, random_seed=42)

        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"
        assert math.isfinite(result.best_metric_value)
        assert result.model_path and Path(result.model_path).exists()
        print(f"  SFT sweep: best_metric={result.best_metric_value:.6f}, model={result.model_path}")


# ---------------------------------------------------------------------------
# S.5  LoRA sweep with peft (HuggingFace model)
# ---------------------------------------------------------------------------

def test_s_5_lora_sweep():
    """LoRA sweep over learning_rate using peft on tiny-gpt2."""
    from serve.sweep_runner import run_sweep

    client = make_client()
    ds_name = ingest_dataset(client, "sweep-lora", _sft_data(10))

    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "lora_data.jsonl"
        write_jsonl(source_path, _sft_data(10))

        config = SweepConfig(
            dataset_name=ds_name,
            output_dir=tmpdir,
            base_output_dir=tmpdir,
            parameters=(
                SweepParameter(name="learning_rate", values=(1e-4, 5e-4)),
            ),
            strategy="grid",
            max_trials=10,
            metric="train_loss",
            minimize=True,
            training_method="lora-train",
            method_args=(
                ("base_model_path", HF_MODEL),
                ("lora_data_path", str(source_path)),
                ("epochs", "1"),
                ("batch_size", "2"),
                ("max_token_length", "64"),
                ("validation_split", "0.0"),
                ("precision_mode", "fp32"),
            ),
        )
        result = run_sweep(client, config, random_seed=42)

        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"
        assert math.isfinite(result.best_metric_value)
        assert result.model_path and Path(result.model_path).exists()
        print(f"  LoRA sweep: best_metric={result.best_metric_value:.6f}")


# ---------------------------------------------------------------------------
# S.6  DPO sweep with trl (HuggingFace model)
# ---------------------------------------------------------------------------

def test_s_6_dpo_sweep():
    """DPO sweep over learning_rate using trl DPOTrainer on tiny-gpt2."""
    from serve.sweep_runner import run_sweep

    client = make_client()
    ds_name = ingest_dataset(client, "sweep-dpo", _dpo_data(10))

    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "dpo_data.jsonl"
        write_jsonl(source_path, _dpo_data(10))

        config = SweepConfig(
            dataset_name=ds_name,
            output_dir=tmpdir,
            base_output_dir=tmpdir,
            parameters=(
                SweepParameter(name="learning_rate", values=(1e-4, 5e-4)),
            ),
            strategy="grid",
            max_trials=10,
            metric="train_loss",
            minimize=True,
            training_method="dpo-train",
            method_args=(
                ("base_model", HF_MODEL),
                ("dpo_data_path", str(source_path)),
                ("epochs", "1"),
                ("batch_size", "2"),
                ("max_token_length", "64"),
                ("validation_split", "0.0"),
                ("precision_mode", "fp32"),
            ),
        )
        result = run_sweep(client, config, random_seed=42)

        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"
        assert math.isfinite(result.best_metric_value)
        print(f"  DPO sweep: best_metric={result.best_metric_value:.6f}")


# ---------------------------------------------------------------------------
# S.7  KTO sweep with trl
# ---------------------------------------------------------------------------

def test_s_7_kto_sweep():
    """KTO sweep over learning_rate using trl KTOTrainer on tiny-gpt2."""
    from serve.sweep_runner import run_sweep

    client = make_client()
    ds_name = ingest_dataset(client, "sweep-kto", _kto_data(10))

    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "kto_data.jsonl"
        write_jsonl(source_path, _kto_data(10))

        config = SweepConfig(
            dataset_name=ds_name,
            output_dir=tmpdir,
            base_output_dir=tmpdir,
            parameters=(
                SweepParameter(name="learning_rate", values=(1e-4, 5e-4)),
            ),
            strategy="grid",
            max_trials=10,
            metric="train_loss",
            minimize=True,
            training_method="kto-train",
            method_args=(
                ("base_model", HF_MODEL),
                ("kto_data_path", str(source_path)),
                ("epochs", "1"),
                ("batch_size", "2"),
                ("max_token_length", "64"),
                ("validation_split", "0.0"),
                ("precision_mode", "fp32"),
            ),
        )
        result = run_sweep(client, config, random_seed=42)

        assert len(result.trials) == 2, f"Expected 2 trials, got {len(result.trials)}"
        assert math.isfinite(result.best_metric_value)
        print(f"  KTO sweep: best_metric={result.best_metric_value:.6f}")


# ---------------------------------------------------------------------------
# S.8  Partial failure handling (some trials fail, others succeed)
# ---------------------------------------------------------------------------

def test_s_8_partial_failure():
    """Sweep where some trials fail — result should contain only successful ones."""
    from serve.sweep_runner import run_sweep

    client = make_client()
    ds_name = ingest_dataset(client, "sweep-partial-fail", _text_data(10))

    with tempfile.TemporaryDirectory() as tmpdir:
        config = SweepConfig(
            dataset_name=ds_name,
            output_dir=tmpdir,
            base_output_dir=tmpdir,
            parameters=(
                # learning_rate=0.0 will cause training to fail (no learning)
                # learning_rate=0.01 should succeed
                SweepParameter(name="learning_rate", values=(0.01, 0.0)),
            ),
            strategy="grid",
            max_trials=10,
            metric="train_loss",
            minimize=True,
            training_method="train",
            method_args=(
                ("hidden_dim", "32"),
                ("num_layers", "1"),
                ("attention_heads", "2"),
                ("epochs", "1"),
                ("batch_size", "2"),
                ("max_token_length", "64"),
                ("precision_mode", "fp32"),
            ),
        )
        result = run_sweep(client, config, random_seed=42)

        # At least one trial should succeed
        assert len(result.trials) >= 1, f"Expected at least 1 successful trial, got {len(result.trials)}"
        assert math.isfinite(result.best_metric_value)
        print(f"  {len(result.trials)} trial(s) succeeded, best_metric={result.best_metric_value:.6f}")


# ---------------------------------------------------------------------------
# S.9  All trials fail → CrucibleSweepError
# ---------------------------------------------------------------------------

def test_s_9_all_trials_fail():
    """Sweep where all trials fail raises CrucibleSweepError."""
    from core.errors import CrucibleSweepError
    from serve.sweep_runner import run_sweep

    client = make_client()
    ds_name = ingest_dataset(client, "sweep-all-fail", _text_data(5))

    with tempfile.TemporaryDirectory() as tmpdir:
        config = SweepConfig(
            dataset_name=ds_name,
            output_dir=tmpdir,
            base_output_dir=tmpdir,
            parameters=(
                SweepParameter(name="learning_rate", values=(0.0,)),
            ),
            strategy="grid",
            max_trials=10,
            metric="train_loss",
            minimize=True,
            training_method="train",
            method_args=(
                ("hidden_dim", "32"),
                ("num_layers", "1"),
                ("attention_heads", "2"),
                ("epochs", "1"),
                ("batch_size", "2"),
                ("max_token_length", "64"),
                ("precision_mode", "fp32"),
                # Bad output dir to guarantee failure
                ("output_dir", "/nonexistent/path/that/cannot/be/created"),
            ),
        )
        try:
            run_sweep(client, config, random_seed=42)
            raise AssertionError("Expected CrucibleSweepError but sweep succeeded")
        except CrucibleSweepError as exc:
            assert "No sweep trials completed" in str(exc)
            assert "First trial error" in str(exc)
            print(f"  CrucibleSweepError raised correctly: {str(exc)[:80]}...")


# ---------------------------------------------------------------------------
# S.10  Model auto-registration after sweep
# ---------------------------------------------------------------------------

def test_s_10_auto_registration():
    """Best trial model is auto-registered in the model registry."""
    from serve.sweep_runner import run_sweep
    from store.model_registry import ModelRegistry

    client = make_client()
    ds_name = ingest_dataset(client, "sweep-autoreg", _text_data(10))

    with tempfile.TemporaryDirectory() as tmpdir:
        config = SweepConfig(
            dataset_name=ds_name,
            output_dir=tmpdir,
            base_output_dir=tmpdir,
            parameters=(
                SweepParameter(name="learning_rate", values=(0.01,)),
            ),
            strategy="grid",
            max_trials=10,
            metric="train_loss",
            minimize=True,
            training_method="train",
            method_args=(
                ("hidden_dim", "32"),
                ("num_layers", "1"),
                ("attention_heads", "2"),
                ("epochs", "1"),
                ("batch_size", "2"),
                ("max_token_length", "64"),
                ("precision_mode", "fp32"),
            ),
        )
        result = run_sweep(client, config, random_seed=42)

        # Check the model was registered
        registry = ModelRegistry(Path(_TMP_ROOT))
        names = registry.list_model_names()
        expected_name = "train_sweep_best"
        assert expected_name in names, f"Model '{expected_name}' not in registry. Found: {names}"
        entry = registry.get_model(expected_name)
        assert entry.model_path == result.model_path, (
            f"Registry path mismatch: {entry.model_path} != {result.model_path}"
        )
        print(f"  Auto-registered '{expected_name}' -> {entry.model_path}")


# ---------------------------------------------------------------------------
# S.11  Maximize direction (pick highest metric)
# ---------------------------------------------------------------------------

def test_s_11_maximize():
    """Sweep with minimize=False picks the trial with the highest metric."""
    from serve.sweep_runner import run_sweep

    client = make_client()
    ds_name = ingest_dataset(client, "sweep-maximize", _text_data(10))

    with tempfile.TemporaryDirectory() as tmpdir:
        config = SweepConfig(
            dataset_name=ds_name,
            output_dir=tmpdir,
            base_output_dir=tmpdir,
            parameters=(
                SweepParameter(name="learning_rate", values=(0.001, 0.01)),
            ),
            strategy="grid",
            max_trials=10,
            metric="train_loss",
            minimize=False,  # Pick highest loss
            training_method="train",
            method_args=(
                ("hidden_dim", "32"),
                ("num_layers", "1"),
                ("attention_heads", "2"),
                ("epochs", "1"),
                ("batch_size", "2"),
                ("max_token_length", "64"),
                ("precision_mode", "fp32"),
            ),
        )
        result = run_sweep(client, config, random_seed=42)

        assert len(result.trials) == 2
        # The best should be the one with HIGHER loss
        metrics = {t.trial_id: t.metric_value for t in result.trials}
        best_metric = metrics[result.best_trial_id]
        other_id = [tid for tid in metrics if tid != result.best_trial_id][0]
        assert best_metric >= metrics[other_id], (
            f"Maximize failed: best={best_metric}, other={metrics[other_id]}"
        )
        print(f"  maximize: best={best_metric:.6f} >= other={metrics[other_id]:.6f}")


# ---------------------------------------------------------------------------
# S.12  CLI --cluster flag parsing (no actual remote, just arg parsing)
# ---------------------------------------------------------------------------

def test_s_12_cli_cluster_flag():
    """Verify the sweep CLI accepts --cluster and resource flags."""
    import argparse
    from cli.sweep_command import add_sweep_command

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    add_sweep_command(sub)

    args = parser.parse_args([
        "sweep",
        "--dataset", "mydata",
        "--output-dir", "/tmp/out",
        "--params", '{"parameters": []}',
        "--method", "sft",
        "--cluster", "my-gpu-cluster",
        "--partition", "a100",
        "--gpus-per-node", "4",
        "--gpu-type", "a100",
        "--memory", "64G",
        "--time-limit", "08:00:00",
    ])

    assert args.cluster == "my-gpu-cluster"
    assert args.partition == "a100"
    assert args.gpus_per_node == 4
    assert args.gpu_type == "a100"
    assert args.memory == "64G"
    assert args.time_limit == "08:00:00"
    print(f"  All cluster flags parsed correctly")


# ---------------------------------------------------------------------------
# S.13  dispatch_training wiring for all sweep-relevant methods
# ---------------------------------------------------------------------------

def test_s_13_dispatch_all_methods():
    """Verify dispatch_training maps all 13 methods without import errors."""
    from core.training_methods import TRAINING_METHOD_DISPATCH, ALL_TRAINING_METHODS

    missing = []
    for method in ALL_TRAINING_METHODS:
        if method not in TRAINING_METHOD_DISPATCH:
            missing.append(method)
        else:
            client_name, opts_class = TRAINING_METHOD_DISPATCH[method]
            # Verify the options class is a real dataclass
            import dataclasses
            if not dataclasses.is_dataclass(opts_class):
                missing.append(f"{method}: {opts_class} is not a dataclass")

    assert not missing, f"Missing or broken dispatch entries: {missing}"
    print(f"  All {len(ALL_TRAINING_METHODS)} methods mapped in TRAINING_METHOD_DISPATCH")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("CRUCIBLE: Hyperparameter Sweep End-to-End Test Suite")
    print(f"Temp data root: {_TMP_ROOT}")
    print("=" * 70)

    tests = [
        ("S.1", "Basic sweep: grid search on .pt model", test_s_1_basic_sweep_grid),
        ("S.2", "Random search strategy", test_s_2_random_strategy),
        ("S.3", "Multi-parameter grid (cartesian product)", test_s_3_multi_param_grid),
        ("S.4", "SFT sweep with trl (HuggingFace)", test_s_4_sft_sweep),
        ("S.5", "LoRA sweep with peft (HuggingFace)", test_s_5_lora_sweep),
        ("S.6", "DPO sweep with trl (HuggingFace)", test_s_6_dpo_sweep),
        ("S.7", "KTO sweep with trl (HuggingFace)", test_s_7_kto_sweep),
        ("S.8", "Partial failure handling", test_s_8_partial_failure),
        ("S.9", "All trials fail raises error", test_s_9_all_trials_fail),
        ("S.10", "Model auto-registration", test_s_10_auto_registration),
        ("S.11", "Maximize direction", test_s_11_maximize),
        ("S.12", "CLI --cluster flag parsing", test_s_12_cli_cluster_flag),
        ("S.13", "dispatch_training wiring for all methods", test_s_13_dispatch_all_methods),
    ]

    for test_id, test_name, fn in tests:
        run_test(test_id, test_name, fn)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"{'ID':<6} | {'STATUS':<6} | {'TIME':<8} | NAME")
    print("-" * 70)
    passed = failed = 0
    for r in RESULTS:
        print(f"{r['id']:<6} | {r['status']:<6} | {r['elapsed']:<8} | {r['name']}")
        if r["status"] == "PASS":
            passed += 1
        else:
            failed += 1
    print("-" * 70)
    print(f"TOTAL: {passed + failed} tests | {passed} PASSED | {failed} FAILED")
    print("=" * 70)

    # Write results to file
    results_path = Path(__file__).resolve().parent / "test_results_sweep.txt"
    with open(results_path, "w") as f:
        f.write("CRUCIBLE: Hyperparameter Sweep E2E Test Results\n")
        f.write(f"Temp data root: {_TMP_ROOT}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'ID':<6} | {'STATUS':<6} | {'TIME':<8} | NAME\n")
        f.write("-" * 70 + "\n")
        for r in RESULTS:
            f.write(f"{r['id']:<6} | {r['status']:<6} | {r['elapsed']:<8} | {r['name']}\n")
        f.write("-" * 70 + "\n")
        f.write(f"TOTAL: {passed + failed} | {passed} PASSED | {failed} FAILED\n")
        f.write("=" * 70 + "\n")

        failures = [r for r in RESULTS if r["status"] == "FAIL"]
        if failures:
            f.write("\n\nDETAILED FAILURES:\n")
            f.write("=" * 70 + "\n")
            for r in failures:
                f.write(f"\n{r['id']} - {r['name']}:\n{r['error']}\n")
    print(f"\nResults written to: {results_path}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
