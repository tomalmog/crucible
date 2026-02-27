"""HellaSwag benchmark implementation.

HellaSwag — evaluates commonsense reasoning by asking the model
to choose the most plausible sentence continuation from 4 options.
Scores by comparing sequence loss (perplexity) across completions.
"""

from __future__ import annotations

from eval.benchmark_runner import BenchmarkResult
from eval.benchmarks._model_loader import compute_sequence_loss, load_eval_model


def run_hellaswag(model_path: str) -> BenchmarkResult:
    """Run HellaSwag benchmark against a model.

    For each example, computes sequence loss for each completion
    option and picks the one with the lowest loss.
    """
    eval_model = load_eval_model(model_path)
    examples = _load_hellaswag_examples()
    correct = 0
    for example in examples:
        losses = [
            compute_sequence_loss(eval_model, example["ctx"] + " " + ending)
            for ending in example["endings"]
        ]
        predicted = int(min(range(len(losses)), key=lambda i: losses[i]))
        if predicted == example["label"]:
            correct += 1
    total = max(len(examples), 1)
    score = round((correct / total) * 100, 2)
    return BenchmarkResult(
        benchmark_name="hellaswag",
        score=score,
        num_examples=len(examples),
        correct=correct,
        details={"format": "multiple_choice", "metric": "accuracy"},
    )


def _load_hellaswag_examples() -> list[dict[str, object]]:
    """Load HellaSwag examples from HuggingFace datasets."""
    from core.errors import ForgeDependencyError
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as error:
        raise ForgeDependencyError(
            "HellaSwag benchmark requires the datasets package. "
            "Install with: pip install datasets"
        ) from error
    ds = load_dataset("Rowan/hellaswag", split="validation")
    return [{"ctx": r["ctx"], "endings": r["endings"],
             "label": int(r["label"])} for r in ds]
