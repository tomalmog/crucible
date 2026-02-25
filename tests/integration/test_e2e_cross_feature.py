"""Integration tests verifying multiple features work together without conflict."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from serve.experiment_tracker import ExperimentTracker
from serve.cost_tracker import CostTracker
from serve.recipe_manager import RecipeManager
from serve.dataset_curator import score_examples
from eval.evaluation_harness import EvaluationHarness
from serve.synthetic_data import (
    generate_synthetic_data,
    filter_by_quality,
    export_synthetic_data,
)


def test_experiment_and_registry_coexist(tmp_path: Path) -> None:
    """ExperimentTracker and CostTracker share data_root without conflict."""
    tracker = ExperimentTracker(tmp_path)
    tracker.log_metrics("run-1", 1, {"loss": 0.5})

    cost = CostTracker(tmp_path)
    cost.log_run_cost("run-1", 3600.0)

    metrics = tracker.get_run_metrics("run-1")
    run_cost = cost.get_run_cost("run-1")

    assert len(metrics) == 1
    assert run_cost is not None
    assert run_cost.gpu_hours == pytest.approx(1.0)


def test_cost_tracking_for_run(tmp_path: Path) -> None:
    """Log cost for a run, retrieve by run_id, verify fields are correct."""
    cost = CostTracker(tmp_path)
    result = cost.log_run_cost(
        "cost-run", 7200.0, gpu_type="a100", tdp_watts=400
    )
    loaded = cost.get_run_cost("cost-run")

    assert loaded is not None
    assert loaded.run_id == "cost-run"
    assert loaded.gpu_hours == pytest.approx(2.0)
    assert loaded.gpu_type == "a100"


def test_recipe_round_trip(tmp_path: Path) -> None:
    """Seed hyperparameters, export recipe, import, and verify round-trip."""
    tracker = ExperimentTracker(tmp_path)
    tracker.log_hyperparameters(
        "recipe-run", {"learning_rate": 0.001, "batch_size": 32}
    )

    manager = RecipeManager(tmp_path)
    export_path = str(tmp_path / "exported.json")
    manager.export_recipe("recipe-run", export_path)
    manager.import_recipe(export_path)

    # Read the exported file to get the recipe name
    with open(export_path) as f:
        exported = json.load(f)
    recipe_name = exported.get("name", "recipe-run")

    recipe = manager.get_recipe(recipe_name)
    assert recipe["hyperparameters"]["learning_rate"] == 0.001
    assert recipe["hyperparameters"]["batch_size"] == 32


def test_curate_identifies_quality() -> None:
    """Score a mix of records and verify quality differentiation."""
    records = [
        {
            "record_id": "good1",
            "text": (
                "A well-written document with substantial content"
                " about machine learning and neural networks."
            ),
        },
        {
            "record_id": "good2",
            "text": (
                "Another quality document discussing the fundamentals"
                " of deep learning architectures."
            ),
        },
        {"record_id": "bad1", "text": "hi"},
        {"record_id": "bad2", "text": ""},
    ]
    scores = score_examples(records)
    high_quality = [s for s in scores if s.score >= 0.8]
    low_quality = [s for s in scores if s.score < 0.8]

    assert len(high_quality) >= 1  # at least the good records
    assert len(low_quality) >= 1  # at least the bad ones


def test_eval_persists_results(tmp_path: Path) -> None:
    """Evaluate, list evaluations, and load evaluation round-trip."""
    harness = EvaluationHarness(tmp_path)
    result = harness.evaluate("model.pt", benchmarks=["mmlu", "gsm8k"])
    assert len(result.benchmark_results) == 2

    eval_ids = harness.list_evaluations()
    assert len(eval_ids) >= 1

    loaded = harness.load_evaluation(eval_ids[0])
    assert loaded["model_path"] == "model.pt"


def test_synthetic_full_pipeline(tmp_path: Path) -> None:
    """Generate, filter, export, and read back synthetic data."""
    examples = generate_synthetic_data(
        ["Tell me about AI", "What is ML?"], count=10
    )
    assert len(examples) == 10

    filtered = filter_by_quality(examples, min_quality=0.3)
    assert all(e.quality_score >= 0.3 for e in filtered)

    output = str(tmp_path / "synthetic.jsonl")
    count = export_synthetic_data(filtered, output)
    assert count == len(filtered)

    # Read back and verify structure
    lines = Path(output).read_text().strip().splitlines()
    assert len(lines) == count
    for line in lines:
        obj = json.loads(line)
        assert "prompt" in obj
        assert "response" in obj
