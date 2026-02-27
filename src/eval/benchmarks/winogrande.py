"""WinoGrande benchmark implementation.

WinoGrande — evaluates commonsense reasoning through
pronoun resolution tasks. Each example has a sentence with
a blank and two candidate completions.
"""

from __future__ import annotations

from eval.benchmark_runner import BenchmarkResult
from eval.benchmarks._model_loader import compute_sequence_loss, load_eval_model


def run_winogrande(model_path: str) -> BenchmarkResult:
    """Run WinoGrande benchmark against a model.

    For each example, computes sequence loss for each option
    filling the blank and picks the lower-loss completion.
    """
    eval_model = load_eval_model(model_path)
    examples = _load_winogrande_examples()
    correct = 0
    for example in examples:
        sentence = example["sentence"]
        option1 = sentence.replace("_", str(example["option1"]))
        option2 = sentence.replace("_", str(example["option2"]))
        loss1 = compute_sequence_loss(eval_model, option1)
        loss2 = compute_sequence_loss(eval_model, option2)
        predicted = 1 if loss1 <= loss2 else 2
        if predicted == example["answer"]:
            correct += 1
    total = max(len(examples), 1)
    score = round((correct / total) * 100, 2)
    return BenchmarkResult(
        benchmark_name="winogrande",
        score=score,
        num_examples=len(examples),
        correct=correct,
        details={"format": "binary_choice", "metric": "accuracy"},
    )


def _load_winogrande_examples() -> list[dict[str, object]]:
    """Load WinoGrande examples from HuggingFace datasets."""
    from core.errors import ForgeDependencyError
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as error:
        raise ForgeDependencyError(
            "WinoGrande benchmark requires the datasets package. "
            "Install with: pip install datasets"
        ) from error
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    return [{"sentence": r["sentence"], "option1": r["option1"],
             "option2": r["option2"], "answer": int(r["answer"])} for r in ds]
